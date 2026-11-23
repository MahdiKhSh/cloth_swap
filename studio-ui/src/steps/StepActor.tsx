import { ReferenceMaskEditor } from "../components/ReferenceMaskEditor";
import clsx from "clsx";
import { BoxSelect, Loader2, MinusCircle, Pause, Play, PlusCircle, ScanSearch, TextSearch, Trash2, UserCheck, Workflow } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, urls, type Polygons, type Prompt } from "../api";
import { FrameViewer, type Layer, type Tool } from "../components/FrameViewer";
import { JobPanel } from "../components/JobPanel";
import { Sam3SettingsPanel } from "../components/Sam3SettingsPanel";
import { Timeline } from "../components/Timeline";
import { useAction, useDebounced, useJob, useKey } from "../hooks";
import { useNum, useT } from "../i18n";
import { StepFooter, useWs } from "../pages/Workspace";
import { Badge, Button, Card, inputClass, SectionTitle, useToasts } from "../ui";
import { usePlayer } from "./usePlayer";

type PromptMap = Record<number, Prompt>;

const toMap = (prompts: Prompt[]): PromptMap => Object.fromEntries(prompts.map((p) => [p.frame, p]));
const isEmpty = (p?: Prompt) => !p || (p.points.length === 0 && !p.box);

export function StepActor() {
  const t = useT();
  const num = useNum();
  const { project: p, refresh, job: lastJob } = useWs();
  const v = p.video!;
  const trim = p.trim!;
  const [prompts, setPrompts] = useState<PromptMap>(() => toMap(p.actor.prompts));
  const initial = p.actor.prompts[0]?.frame ?? p.keyframe.frame ?? trim.start;
  const player = usePlayer(trim.start, trim.end, v.fps!, initial);
  const frame = player.frame;
  const [tool, setTool] = useState<Tool>("add");
  const [maskEditorOpen, setMaskEditorOpen] = useState(false);
  const [maskPart, setMaskPart] = useState<string>("actor");
  const [layerJobId, setLayerJobId] = useState<string | null>(null);
  const [people, setPeople] = useState<{ frame: number; list: { box: number[]; polygons: Polygons }[] } | null>(null);
  const [preview, setPreview] = useState<{ key: string; polygons: Polygons } | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [jobId, setJobId] = useState<string | null>(() => {
    const j = lastJob("track");
    return j && (j.status === "running" || j.status === "queued") ? j.id : null;
  });
  // Show the composed edit region immediately when the manual mask job finishes.
  useEffect(() => { setMaskPart(p.mask_layers.ready ? "edit" : "actor"); }, [p.mask_layers.ready]);
  const masks = useRef(new Map<string, Polygons>());
  const maskKey = `${p.updated}:${maskPart}:${frame}`;
  const [, force] = useState(0);
  const push = useToasts((s) => s.push);
  const detect = useAction();
  const action = useAction();

  const current = prompts[frame];
  const visible = useMemo(() => new Set(p.actor.visible), [p.actor.visible]);

  // save selections (debounced)
  const saved = useDebounced(prompts, 400);
  const first = useRef(true);
  useEffect(() => {
    if (first.current) { first.current = false; return; }
    api.setPrompts(p.id, Object.values(saved).filter((x) => !isEmpty(x)));
  }, [saved, p.id]);

  // live preview of the selection on this frame
  const key = JSON.stringify(current ?? null);
  const debouncedKey = useDebounced(key, 120);
  useEffect(() => {
    if (isEmpty(current) || debouncedKey !== key) return;
    let stop = false;
    setPreviewing(true);
    action.run(() => api.segment(p.id, current!)).then((r) => {
      if (!stop && r) setPreview({ key, polygons: r.polygons });
    }).finally(() => !stop && setPreviewing(false));
    return () => { stop = true; };
  }, [debouncedKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // tracked mask of this frame
  useEffect(() => {
    if (!p.actor.tracked || (maskPart !== "actor" && !p.mask_layers.ready) || masks.current.has(maskKey)) return;
    let stop = false;
    api.actorMask(p.id, frame, maskPart).then((r) => {
      masks.current.set(maskKey, r.polygons);
      if (!stop) force((x) => x + 1);
    });
    return () => { stop = true; };
  }, [frame, maskPart, p.actor.tracked, p.mask_layers.ready, p.actor.trim?.[0], p.updated]); // eslint-disable-line react-hooks/exhaustive-deps

  const job = useJob(jobId, async () => {
    masks.current.clear();
    setMaskPart("actor");
    setDirty(false);
    await refresh();
  }, () => setJobId(null));
  const layerJob = useJob(layerJobId, async () => {
    masks.current.clear();
    setMaskPart("edit");
    await refresh();
  }, (completed) => {
    if (completed.status === "failed") push(completed.error ?? "ساخت ماسک نهایی ناموفق بود", "error");
    setLayerJobId(null);
  });

  const edit = (fn: (pr: Prompt) => Prompt) => {
    
    setPrompts((m) => {
      const base = m[frame] ?? { frame, points: [], labels: [], box: null };
      const next = fn(base);
      const copy = { ...m };
      if (isEmpty(next)) delete copy[frame]; else copy[frame] = next;
      return copy;
    });
    if (p.actor.tracked) setDirty(true);
    setPeople(null);
  };

  const [query, setQuery] = useState("");
  const runDetect = async (text?: string) => {
    if (text && /[؀-ۿ]/.test(text)) {
      push(t("english_only"), "error");
      return;
    }
    const r = await detect.run(() => api.detect(p.id, frame, text));
    if (r) setPeople({ frame, list: r.people });
  };

  const track = async () => {
    await api.setPrompts(p.id, Object.values(prompts).filter((x) => !isEmpty(x)));
    const j = await action.run(() => api.track(p.id));
    if (j) setJobId(j.id);
  };

  useKey((e) => {
    if (maskEditorOpen) return;
    if (e.key === "ArrowRight") player.setFrame(Math.min(trim.end - 1, frame + 1));
    else if (e.key === "ArrowLeft") player.setFrame(Math.max(trim.start, frame - 1));
    else if (e.key === " ") { e.preventDefault(); player.setPlaying(!player.playing); }
    else if (e.key === "a") setTool("add");
    else if (e.key === "r") setTool("remove");
    else if (e.key === "b") setTool("box");
    else if (e.key === "Delete") edit(() => ({ frame, points: [], labels: [], box: null }));
  }, [frame, player.playing, trim.start, trim.end, prompts, maskEditorOpen]);

  const layers: Layer[] = [];
  const showPeople = !!(people && people.frame === frame);
  if (maskPart !== "actor" && p.mask_layers.ready && masks.current.has(maskKey)) {
    // The manually tracked part or composed edit/protect mask takes precedence.
    layers.push({polygons: masks.current.get(maskKey)!, className: (() => {
      const op = p.mask_layers.rules.find(r => `layer:${r.name}` === maskPart)?.operation;
      return maskPart === "edit" || op === "add" ? "mask-fill" : op === "subtract" ? "mask-protect" : "mask-person";
    })()});
  } else if (showPeople) {
    people!.list.forEach((person, i) => layers.push({
      key: `p${i}`, polygons: person.polygons, className: "mask-person",
      onClick: () => edit(() => ({frame, points: [], labels: [], box: person.box})),
    }));
  } else if (!isEmpty(current) && preview?.key === key) {
    layers.push({polygons: preview.polygons, className: "mask-fill"});
  } else if (p.actor.tracked && masks.current.has(maskKey)) {
    layers.push({polygons: masks.current.get(maskKey)!, className: "mask-tracked"});
  }

  const running = !!jobId && (!job || job.status === "running" || job.status === "queued");
  const layersRunning = !!layerJobId && (!layerJob || layerJob.status === "running" || layerJob.status === "queued");
  const promptFrames = Object.keys(prompts).map(Number).sort((a, b) => a - b);
  const tools: { id: Tool; icon: typeof PlusCircle; label: string; tone: string }[] = [
    { id: "add", icon: PlusCircle, label: t("toolAdd"), tone: "text-emerald-300" },
    { id: "remove", icon: MinusCircle, label: t("toolRemove"), tone: "text-rose-300" },
    { id: "box", icon: BoxSelect, label: t("toolBox"), tone: "text-violet-300" },
  ];

  return (
    <div className="flex h-full flex-col gap-4">
      <SectionTitle title={t("actorTitle")} hint={t("actorHint")}
        right={p.actor.tracked && <Badge tone="ok"><UserCheck className="size-3.5" /> {t("visibleIn", { n: num(p.actor.visible.length), m: num(trim.end - trim.start) })}</Badge>} />
      <div className="flex min-h-0 flex-1 gap-5">
        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex rounded-xl border border-white/10 bg-black/30 p-1">
              {tools.map((tl) => (
                <button key={tl.id} onClick={() => { setTool(tl.id); setPeople(null); }}
                  className={clsx("flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm transition",
                    tool === tl.id && !showPeople ? "bg-white/12 text-white" : "text-zinc-400 hover:text-zinc-200")}>
                  <tl.icon className={clsx("size-4", tl.tone)} /> {tl.label}
                </button>
              ))}
            </div>
            <Button variant="secondary" icon={<ScanSearch className="size-4" />} loading={detect.busy} onClick={() => runDetect()}>
              {detect.busy ? t("detecting") : t("detect")}
            </Button>
            <form className="flex min-w-72 flex-1 items-center gap-2" title={t("findHint")}
              onSubmit={(e) => { e.preventDefault(); if (query.trim()) runDetect(query.trim()); }}>
              <div className="relative flex-1">
                <TextSearch className="pointer-events-none absolute start-3 top-1/2 size-4 -translate-y-1/2 text-zinc-500" />
                <input dir="ltr" value={query} onChange={(e) => setQuery(e.target.value)} placeholder={t("findPh")}
                  className={clsx(inputClass, "h-10 ps-9")} aria-label={t("findByText")} />
              </div>
              <Button type="submit" variant="secondary" disabled={!query.trim() || detect.busy}>{t("find")}</Button>
            </form>
            {!isEmpty(current) && (
              <Button variant="ghost" icon={<Trash2 className="size-4" />} onClick={() => edit(() => ({ frame, points: [], labels: [], box: null }))}>
                {t("clearFrame")}
              </Button>
            )}
            <span className="ms-auto flex items-center gap-2 text-xs text-zinc-500">
              {previewing && <Loader2 className="size-3.5 animate-spin text-violet-300" />}
              {showPeople ? (people!.list.length ? t("pickPerson") : t("noPeople")) : null}
            </span>
          </div>
          <p className="text-xs text-zinc-400">۱. کل شخص را ردیابی کن. ۲. هر تعداد آبجکت دلخواه را روی شخص جداشده با پرامپت، نقطه یا کادر انتخاب کن. ۳. برای هر آبجکت تعیین کن در ماسک نهایی با + اضافه شود، با − کم شود یا نادیده گرفته شود.</p>
          {p.actor.tracked && <div className="flex flex-wrap items-center gap-2 text-xs text-zinc-300">
            <span>نمایش ماسک‌های ردیابی‌شده:</span>
            <select className={clsx(inputClass, "!w-auto")} value={maskPart} onChange={e => setMaskPart(e.target.value)}>
              <option value="actor">کل شخص (ماسک مثبت پایه)</option>
              <option value="edit" disabled={!p.mask_layers.ready}>ناحیه قابل‌تغییر نهایی (+)</option>
              <option value="protect" disabled={!p.mask_layers.ready}>ناحیه محافظت‌شده (−)</option>
              <option value="removed" disabled={!p.mask_layers.ready}>پیکسل‌های واقعاً کم‌شده از بازیگر (Actor − Edit)</option>
              {p.mask_layers.ready && p.mask_layers.rules.map(r => <option key={r.name} value={`layer:${r.name}`}>{r.operation === "add" ? "+" : r.operation === "subtract" ? "−" : "·"} {r.name}</option>)}
            </select>
          </div>}
          <FrameViewer className="min-h-0 flex-1" src={urls.frame(p.id, frame)} width={v.width!} height={v.height!}
            layers={layers} tool={showPeople || maskPart !== "actor" ? null : tool}
            points={(current?.points ?? []).map(([x, y], i) => ({x, y, label: current?.labels[i] ?? 1}))}
            box={current?.box ?? null}
            onPoint={(x, y, label) => edit(pr => ({...pr, points: [...pr.points, [x,y]], labels: [...pr.labels, label]}))}
            onBox={box => edit(pr => ({...pr, box}))} />
          <div className="flex items-center gap-3">
            <Button variant="secondary" className="size-10 !px-0" onClick={() => player.setPlaying(!player.playing)}>
              {player.playing ? <Pause className="size-4" /> : <Play className="size-4" />}
            </Button>
            <div className="flex-1">
              <Timeline count={v.num_frames!} current={frame} onSeek={(f) => { player.setPlaying(false); player.setFrame(f); }}
                thumb={(f) => urls.thumb(p.id, f)} trim={trim} limit={trim}
                coverage={p.actor.tracked ? visible : null} markers={promptFrames} />
            </div>
            <span className="w-16 text-end text-xs tabular-nums text-zinc-400" dir="ltr">#{frame}</span>
          </div>
          <p className="text-[11px] text-zinc-600">{t("shortcuts")}</p>
          <p className="text-[11px] text-zinc-600" dir="ltr">🔎 {t("findHint")}</p>
        </div>

        <aside className="flex w-80 shrink-0 flex-col gap-4 overflow-y-auto">
          <Sam3SettingsPanel projectId={p.id} onUpdated={refresh} disabled={running || layersRunning || dirty} />
          <Card className="p-4">
            <div className="mb-3 flex items-center justify-between">
              <span className="text-sm font-semibold text-white">{t("selections")}</span>
              <Badge>{num(promptFrames.length)}</Badge>
            </div>
            {promptFrames.length === 0 ? (
              <p className="text-xs leading-6 text-zinc-500">{t("noSelections")}<br />{t("modelLoading")}</p>
            ) : (
              <ul className="space-y-2">
                {promptFrames.map((f) => (
                  <li key={f}>
                    <button onClick={() => player.setFrame(f)}
                      className={clsx("flex w-full items-center gap-3 rounded-xl p-1.5 text-start transition",
                        f === frame ? "bg-violet-500/15 ring-1 ring-violet-400/40" : "hover:bg-white/[0.05]")}>
                      <img src={urls.thumb(p.id, f)} className="h-10 w-16 rounded-lg object-cover" />
                      <span className="flex-1 text-xs text-zinc-300">
                        {t("frame")} <span className="tabular-nums">{num(f)}</span>
                        <span className="mt-0.5 block text-[11px] text-zinc-500" dir="ltr">
                          +{prompts[f].labels.filter((l) => l === 1).length} −{prompts[f].labels.filter((l) => l === 0).length}{prompts[f].box ? " ▢" : ""}
                        </span>
                      </span>
                      <span role="button" onClick={(e) => { e.stopPropagation(); setPrompts((m) => { const c = { ...m }; delete c[f]; return c; }); setDirty(true); }}
                        className="rounded-lg p-1.5 text-zinc-500 hover:bg-rose-500/10 hover:text-rose-300">
                        <Trash2 className="size-3.5" />
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Card>

          {dirty && <div className="rounded-xl border border-amber-500/25 bg-amber-500/[0.07] p-3 text-xs text-amber-200">{t("needsRetrack")}</div>}
          <Button variant="primary" size="lg" icon={<Workflow className="size-5" />} loading={running}
            disabled={promptFrames.length === 0 || running || layersRunning} onClick={track}>
            {p.actor.tracked ? "۱. ردیابی مجدد کل شخص" : "۱. ردیابی و سگمنت کل شخص"}
          </Button>
          <Button variant="secondary" size="lg" onClick={() => setMaskEditorOpen(true)}
            disabled={!p.actor.tracked || running || layersRunning || dirty}>
            {p.mask_layers.rules.length ? `۲. مدیریت ${p.mask_layers.rules.length} آبجکت ماسک` : "۲. ساخت آبجکت‌های ماسک و ترکیب نهایی"}
          </Button>
          {p.actor.tracked && <p className="text-xs text-emerald-300">{(p.mask_layers.base_actor ?? true) ? "actor پایه فعال است." : "actor پایه غیرفعال است."} {p.mask_layers.rules.length ? `آبجکت‌ها: ${p.mask_layers.rules.map(r => `${r.operation === "add" ? "+" : r.operation === "subtract" ? "−" : "·"}${r.name}`).join("، ")}` : "هنوز آبجکتی تعریف نشده است."}</p>}
          {job && <JobPanel job={job} />}
          {layerJob && <JobPanel job={layerJob} />}
          {p.actor.tracked && !p.mask_layers.ready && !layersRunning && <p className="text-xs text-amber-300">برای ادامه وارد ویرایشگر ماسک شو، آبجکت‌های دلخواه را بساز و فرمول نهایی را تأیید کن. سگمنت با پرامپت متنی انگلیسی هم فعال است.</p>}
        </aside>
      </div>
      {maskEditorOpen && <ReferenceMaskEditor project={p} preTrackFrame={frame} onClose={() => setMaskEditorOpen(false)} onRefresh={refresh} onBuildStart={setLayerJobId} />}
      <StepFooter next="garment" nextDisabled={!p.actor.tracked || !p.mask_layers.ready || running || layersRunning || dirty} />
    </div>
  );
}
