import { useEffect, useMemo, useState } from "react";
import { Check, PlusCircle, Trash2, X } from "lucide-react";
import { api, urls, type MaskLayer, type Polygons, type Prompt, type Project } from "../api";
import { FrameViewer, type Tool } from "./FrameViewer";
import { useDebounced, useJob } from "../hooks";
import { Button, inputClass } from "../ui";
import { JobPanel } from "./JobPanel";

type ReferenceSelection = Prompt & { candidate: string | null };
const empty = (frame: number, candidate: string | null): ReferenceSelection => ({ frame, candidate, text: "", points: [], labels: [], box: null });

export function ReferenceMaskEditor({ project: p, onClose, onRefresh, preTrackFrame, onBuildStart }: {
  project: Project; onClose: () => void; onRefresh: () => Promise<unknown>; preTrackFrame?: number;
  onBuildStart?: (jobId: string) => void;
}) {
  const fromVideoFrame = preTrackFrame !== undefined;
  const candidate = fromVideoFrame ? null : p.keyframe.chosen!;
  const frame = fromVideoFrame ? (p.mask_layers.rules[0]?.selection?.frame ?? preTrackFrame!) : p.keyframe.frame!;
  const [rules, setRules] = useState<MaskLayer[]>(() => p.mask_layers.rules);
  const [baseActor, setBaseActor] = useState<boolean>(p.mask_layers.base_actor ?? true);
  const [active, setActive] = useState<number | null>(p.mask_layers.rules.length ? 0 : null);
  const [tool, setTool] = useState<Tool | null>("add");
  const [previewCache, setPreviewCache] = useState<Record<string, Polygons>>({});
  const [error, setError] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [actorPolygons, setActorPolygons] = useState<Polygons>([]);

  useEffect(() => {
    if (!fromVideoFrame || !p.actor.tracked) return;
    let canceled = false;
    api.actorMask(p.id, frame, "actor").then(r => {
      if (!canceled) setActorPolygons(r.polygons);
    }).catch(err => { if (!canceled) setError(`ماسک پایه شخص: ${String(err.message ?? err)}`); });
    return () => { canceled = true; };
  }, [p.id, frame, fromVideoFrame, p.actor.tracked]);

  const job = useJob(jobId, async () => { setDirty(false); await onRefresh(); onClose(); }, () => setJobId(null));
  const activeRule = active === null ? undefined : rules[active];
  const selection = (activeRule?.selection?.candidate ?? null) === candidate && activeRule?.selection?.frame === frame ? activeRule.selection : undefined;
  const selectionKey = JSON.stringify(selection ?? null);
  const [previewing, setPreviewing] = useState(false);
  const delayed = useDebounced(selectionKey, 200);

  useEffect(() => {
    if (!selection || delayed !== selectionKey || (!selection.text?.trim() && !selection.points.length && !selection.box)) { setPreviewing(false); return; }
    let canceled = false;
    setPreviewing(true);
    api.segment(p.id, { ...selection, isolated: fromVideoFrame && p.actor.tracked }).then((v) => {
      if (!canceled) {
        setPreviewCache(old => ({ ...old, [selectionKey]: v.polygons }));
        setError(v.area > 0 ? "" : "ماسکی پیدا نشد؛ پرامپت یا انتخاب را اصلاح کن.");
      }
    }).catch((err) => {
      if (!canceled) setError(`پیش‌نمایش ماسک: ${String(err.message ?? err)}`);
    }).finally(() => { if (!canceled) setPreviewing(false); });
    return () => { canceled = true; };
  }, [p.id, delayed, selectionKey, fromVideoFrame, p.actor.tracked, selection]);

  const update = (fn: (old: MaskLayer[]) => MaskLayer[]) => {
    if (jobId) return;
    setRules(fn);
    setDirty(true);
  };
  const setSelection = (fn: (prev: ReferenceSelection) => ReferenceSelection) => {
    if (active === null) return;
    const previous = selection ? { ...selection, candidate } : empty(frame, candidate);
    update((old) => old.map((r, i) => i === active ? { ...r, prompt: "", selection: fn(previous) } : r));
  };

  const valid = useMemo(() => rules.every(r => {
    const s = r.selection;
    return !!(r.name.trim() && s && (s.candidate ?? null) === candidate && s.frame === frame && !!(s.text?.trim() || s.labels.includes(1) || s.box));
  }), [rules, candidate, frame]);

  const build = async () => {
    setError("");
    try {
      await api.setMaskLayers(p.id, rules, baseActor);
      const j = await api.buildMaskLayers(p.id);
      onBuildStart?.(j.id);
      setJobId(j.id);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  };

  const add = () => {
    const name = `obj_${Math.random().toString(36).slice(2, 8)}`;
    update(old => [...old, { name, prompt: "", operation: "ignore", selection: empty(frame, candidate) }]);
    setActive(rules.length);
    setTool("add");
  };

  const expression = `${baseActor ? "actor" : "∅"}${rules.length ? " " : ""}${rules.map(r => r.operation === "add" ? `+ ${r.name}` : r.operation === "subtract" ? `− ${r.name}` : `· ${r.name}`).join(" ")}`.trim() || (baseActor ? "actor" : "∅");

  return <div className="fixed inset-0 z-[100] flex flex-col bg-[#09090f] p-3 sm:p-6" dir="rtl">
    <header className="mb-3 flex shrink-0 items-center justify-between gap-3">
      <div>
        <h2 className="text-lg font-bold text-white">ساخت آبجکت‌های ماسک و ترکیب نهایی</h2>
        <p className="mt-1 text-xs text-zinc-400">هر آبجکت را با پرامپت انگلیسی، نقطه مثبت/منفی یا کادر تعریف کن. بعداً برای هر آبجکت مشخص می‌کنی در ماسک نهایی با + اضافه شود، با − کم شود یا فعلاً نادیده گرفته شود.</p>
      </div>
      <button onClick={onClose} aria-label="بستن" className="rounded-xl bg-white/10 p-3 text-white"><X className="size-5" /></button>
    </header>
    <div className="flex min-h-0 flex-1 flex-col gap-4 lg:flex-row">
      <section className="flex min-h-[240px] min-w-0 flex-1 flex-col rounded-2xl border border-white/10 bg-black p-2">
        <div className="mb-2 flex flex-wrap gap-2">
          {(["add", "remove", "box"] as const).map(v => <button key={v} onClick={() => {
            setTool(v);
            setSelection(s => ({ ...s, text: "", points: v === "box" ? [] : s.points, labels: v === "box" ? [] : s.labels, box: v === "box" ? s.box : null }));
          }}
            className={`rounded-lg border px-3 py-2 text-sm ${tool === v ? "border-violet-400 bg-violet-500/25 text-white" : "border-white/10 text-zinc-400"}`}>
            {v === "add" ? "نقطه مثبت" : v === "remove" ? "نقطه منفی" : "انتخاب کادر"}
          </button>)}
          <button className={`rounded-lg border px-3 py-2 text-sm ${tool === null ? "border-violet-400 text-white" : "border-white/10 text-zinc-400"}`}
            onClick={() => { setTool(null); setSelection(s => ({ ...s, points: [], labels: [], box: null })); }}>پرامپت متنی</button>
          <span className="self-center text-xs text-zinc-500">اول آبجکت را بساز و انتخابش کن، بعد نوع انتخاب آن را مشخص کن.</span>
        </div>
        {active !== null && tool === null && <input className={inputClass} dir="ltr" aria-label="پرامپت ماسک" placeholder="shirt / head / shoes / hand" value={selection?.text ?? ""} onChange={e => setSelection(s => ({ ...s, text: e.target.value, points: [], labels: [], box: null }))} />}
        {previewing && <p className="text-xs text-violet-300">در حال محاسبه پیش‌نمایش SAM3…</p>}
        <FrameViewer className="min-h-0 flex-1"
          src={fromVideoFrame && p.actor.tracked ? urls.isolated(p.id, frame) : fromVideoFrame ? urls.frame(p.id, frame) : urls.candidate(p.id, candidate!)}
          width={p.video!.width!} height={p.video!.height!}
          tool={active === null ? null : tool}
          layers={[
            ...(actorPolygons.length && baseActor ? [{ key: "actor-base", polygons: actorPolygons, className: "mask-fill" }] : []),
            ...rules.flatMap((r, i) => {
              const current = (r.selection?.candidate ?? null) === candidate && r.selection?.frame === frame ? JSON.stringify(r.selection) : null;
              const polygons = current ? previewCache[current] : undefined;
              if (!polygons) return [];
              const className = r.operation === "add" ? "mask-fill" : r.operation === "subtract" ? "mask-protect" : "mask-person";
              return [{ key: `${i}`, polygons, className }];
            }),
          ]}
          points={(selection?.points ?? []).map((point: number[], i: number) => ({ x: point[0], y: point[1], label: selection?.labels[i] ?? 1 }))}
          box={selection?.box}
          onPoint={(x, y, label) => setSelection(s => ({ ...s, points: [...s.points, [x, y]], labels: [...s.labels, label] }))}
          onBox={box => setSelection(s => ({ ...s, box }))} />
      </section>

      <aside className="flex max-h-[42vh] w-full shrink-0 flex-col gap-3 overflow-y-auto rounded-2xl border border-white/10 bg-[#161620] p-3 lg:max-h-none lg:w-[390px]">
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="secondary" onClick={add} icon={<PlusCircle className="size-4" />}>افزودن آبجکت</Button>
          <label className="ms-auto flex items-center gap-2 text-sm text-zinc-200">
            <input type="checkbox" checked={baseActor} onChange={e => { setBaseActor(e.target.checked); setDirty(true); }} />
            کل بازیگر پایه مثبت باشد
          </label>
        </div>

        {rules.map((r, i) => <div key={i} className={`rounded-xl border p-2 ${active === i ? "border-violet-400 bg-violet-500/10" : "border-white/10"}`}>
          <div className="flex items-center gap-2">
            <input className={`${inputClass} min-w-0 flex-1`} dir="ltr" value={r.name} aria-label="اسم آبجکت"
              onChange={e => update(x => x.map((z, j) => j === i ? { ...z, name: e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, "") } : z))} />
            <select className={`${inputClass} !w-auto`} value={r.operation}
              onChange={e => update(x => x.map((z, j) => j === i ? { ...z, operation: e.target.value as MaskLayer["operation"] } : z))}>
              <option value="ignore">ignore</option>
              <option value="add">+ add</option>
              <option value="subtract">- subtract</option>
            </select>
            <button title="حذف آبجکت" className="text-rose-300" onClick={() => { update(x => x.filter((_, j) => j !== i)); setActive(null); }}><Trash2 className="size-4" /></button>
          </div>
          <button className={`mt-2 w-full rounded-lg px-3 py-2 text-sm ${active === i ? "bg-violet-500 text-white" : "bg-white/10 text-zinc-200"}`}
            onClick={() => { setActive(i); setTool(r.selection?.text ? null : r.selection?.box ? "box" : "add"); }}>
            {active === i ? "✓ در حال انتخاب این آبجکت" : "انتخاب / اصلاح این آبجکت"}
          </button>
          {r.selection && <button className="mt-1 text-xs text-rose-300" onClick={() => { update(x => x.map((z, j) => j === i ? { ...z, selection: null } : z)); }}>پاک‌کردن انتخاب</button>}
        </div>)}

        <div className="mt-auto space-y-2 border-t border-white/10 pt-3">
          <p className="text-sm font-medium text-emerald-300">فرمول فعلی</p>
          <p className="text-sm text-zinc-200" dir="ltr">{expression}</p>
          <p className="text-xs leading-5 text-amber-200">اگر فقط علامت‌های add / subtract / ignore یا گزینهٔ actor-base را عوض کنی، مرحلهٔ بعدی بدون ترکینگ دوباره فقط ماسک نهایی را می‌سازد. اگر خود انتخاب آبجکت‌ها را عوض کنی، SAM3 دوباره همان آبجکت‌ها را ترک می‌کند.</p>
          {p.mask_layers.tracked && <p className="text-xs text-emerald-300">آبجکت‌های ترک‌شده از اجرای قبلی موجود هستند و برای recomposition دوباره قابل استفاده‌اند.</p>}
          {error && <p role="alert" className="text-xs text-rose-300">{error}</p>}
          {dirty && <p className="text-xs text-amber-300">تغییرات هنوز روی ماسک ویدئو اعمال نشده‌اند.</p>}
          {job && <JobPanel job={job} />}
          <Button variant="primary" loading={!!jobId} disabled={!valid || !!jobId || previewing} onClick={build} icon={<Check className="size-4" />}>
            {p.mask_layers.tracked ? "ردیابیِ لازم / بازترکیب و ساخت ماسک نهایی" : "ردیابی آبجکت‌ها و ساخت ماسک نهایی"}
          </Button>
          {p.mask_layers.ready && !dirty && <p className="text-xs text-emerald-300">ماسک نهایی قبلاً ساخته شده است؛ در صورت تغییر دوباره بساز.</p>}
        </div>
      </aside>
    </div>
  </div>;
}
