import clsx from "clsx";
import { AlertTriangle, CheckCircle2, ChevronDown, Dice5, ImageUp, Sparkles, Star, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, urls, type Candidate, type Polygons } from "../api";
import { CompareImage } from "../components/Compare";
import { FrameViewer } from "../components/FrameViewer";
import { JobPanel } from "../components/JobPanel";
import { Timeline } from "../components/Timeline";
import { useAction, useDebounced, useJob } from "../hooks";
import { useLang, useNum, useT } from "../i18n";
import { StepFooter, useWs } from "../pages/Workspace";
import { Badge, Button, Card, EmptyState, Field, inputClass, SectionTitle, Segmented } from "../ui";

export function StepKeyframe() {
  const t = useT();
  const fa = useLang((s) => s.lang) === "fa";
  const num = useNum();
  const { project: p, setProject, refresh, job: lastJob } = useWs();
  const v = p.video!;
  const trim = p.trim!;
  const visible = useMemo(() => new Set(p.actor.visible), [p.actor.visible]);
  const [frame, setFrame] = useState<number>(p.keyframe.frame ?? p.keyframe.suggestions[0]?.frame ?? p.actor.visible[0] ?? trim.start);
  const [count, setCount] = useState(1);
  const [advanced, setAdvanced] = useState(false);
  const [extra, setExtra] = useState(p.keyframe.extra || "");
  const [builtIn, setBuiltIn] = useState("");
  const extraDebounced = useDebounced(extra, 400);
  useEffect(() => {
    if (!advanced) return;
    api.instruction(p.id, extraDebounced).then((r) => setBuiltIn(r.text)).catch(() => {});
  }, [advanced, extraDebounced, p.id]);
  const [seed, setSeed] = useState<string>("");
  const [focus, setFocus] = useState<string | null>(p.keyframe.chosen);
  const [outline, setOutline] = useState<Polygons>([]);
  const [jobId, setJobId] = useState<string | null>(() => {
    const j = lastJob("keyframe");
    return j && (j.status === "running" || j.status === "queued") ? j.id : null;
  });
  const fileInput = useRef<HTMLInputElement>(null);
  const action = useAction();

  useEffect(() => {
    let stop = false;
    api.actorMask(p.id, frame).then((r) => !stop && setOutline(r.polygons));
    return () => { stop = true; };
  }, [frame, p.id]);

  const job = useJob(jobId, async (j) => {
    await refresh();
    const made = j.result?.candidates;
    if (made?.length) setFocus(made[0]);
  }, () => setJobId(null));

  const running = !!jobId && (!job || job.status === "running" || job.status === "queued");
  const candidates = [...p.keyframe.candidates].reverse();
  const focused = candidates.find((c) => c.id === focus);
  const canUse = visible.has(frame);

  const generate = async () => {
    const j = await action.run(() => api.generateKeyframes(p.id, {
      frame, count, extra: extra.trim() || undefined, seed: seed ? Number(seed) : undefined,
    }));
    if (j) { setJobId(j.id); setFocus(null); }
  };
  const uploadEdit = async (file: File) => {
    const j = await action.run(() => api.uploadKeyframe(p.id, frame, file));
    if (j) setJobId(j.id);
  };
  const choose = async (c: Candidate) => {
    const res = await action.run(() => api.chooseKeyframe(p.id, c.id));
    if (res) setProject(res);
  };

  return (
    <div className="flex h-full flex-col gap-4">
      <SectionTitle title={t("keyframeTitle")} hint={t("keyframeHint")} />
      <div className="flex min-h-0 flex-1 gap-5">
        <div className="flex min-w-0 flex-1 flex-col gap-3">
          <div className="flex items-center gap-2 overflow-x-auto pb-1">
            <span className="flex shrink-0 items-center gap-1 text-xs text-zinc-500"><Star className="size-3.5 text-amber-300" /> {t("suggested")}</span>
            {p.keyframe.suggestions.map((s) => (
              <button key={s.frame} onClick={() => { setFrame(s.frame); setFocus(null); }}
                className={clsx("relative shrink-0 overflow-hidden rounded-xl ring-2 transition",
                  frame === s.frame && !focus ? "ring-violet-400" : "ring-transparent hover:ring-white/20")}>
                <img src={urls.thumb(p.id, s.frame)} className="h-14 w-24 object-cover" />
                <span className="absolute bottom-1 start-1 rounded bg-black/70 px-1.5 text-[10px] tabular-nums text-white">#{num(s.frame)}</span>
              </button>
            ))}
          </div>

          <div className="min-h-0 flex-1">
            {focused ? (
              <CompareImage key={focused.id} before={urls.frame(p.id, focused.frame)} after={urls.candidate(p.id, focused.id)}
                width={v.width!} height={v.height!} labels={[t("original"), t("edited")]} />
            ) : (
              <FrameViewer className="h-full" src={urls.frame(p.id, frame)} width={v.width!} height={v.height!}
                layers={[{ polygons: outline, className: "mask-tracked" }]}>
                {!canUse && (
                  <div className="absolute inset-x-0 bottom-4 mx-auto w-fit rounded-full bg-black/75 px-4 py-1.5 text-xs text-amber-200">
                    {t("actorNotVisible")}
                  </div>
                )}
              </FrameViewer>
            )}
          </div>
          {!focused && (
            <Timeline count={v.num_frames!} current={frame} onSeek={(f) => setFrame(f)} thumb={(f) => urls.thumb(p.id, f)}
              trim={trim} limit={trim} coverage={visible} markers={p.keyframe.suggestions.map((s) => s.frame)} />
          )}
        </div>

        <aside className="flex w-80 shrink-0 flex-col gap-4 overflow-y-auto">
          <Card className="space-y-4 p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-zinc-300">{t("frame")}</span>
              <Badge tone="brand">#{num(frame)}</Badge>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm text-zinc-300">{t("variants")}</span>
              <Segmented value={count} onChange={setCount} options={[1, 2, 3, 4].map((n) => ({ value: n, label: num(n) }))} />
            </div>
            <p className="text-xs text-zinc-500">{fa ? "برای آزمایش سریع ابتدا یک تصویر بساز؛ در صورت نیاز تعداد را افزایش بده." : "Start with one candidate for faster tests; add variants only when needed."}</p>
            <button className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300" onClick={() => setAdvanced(!advanced)}>
              <ChevronDown className={clsx("size-3.5 transition", advanced && "rotate-180")} /> {t("advanced")}
            </button>
            {advanced && (
              <div className="space-y-3 animate-fade-in">
                <Field label={t("extraInstr")} hint={t("extraHint")}>
                  <textarea dir="ltr" className={clsx(inputClass, "min-h-20 text-xs leading-5")} value={extra}
                    placeholder={t("extraPh")} onChange={(e) => setExtra(e.target.value)} />
                </Field>
                {builtIn && (
                  <details className="rounded-xl border border-white/10 bg-black/20 p-3">
                    <summary className="cursor-pointer text-xs text-zinc-400">{t("builtIn")}</summary>
                    <p dir="ltr" className="mt-2 text-[11px] leading-5 text-zinc-500">{builtIn}</p>
                  </details>
                )}
                <Field label={t("seed")}>
                  <div className="flex gap-2">
                    <input dir="ltr" className={inputClass} value={seed} placeholder={t("random")} inputMode="numeric"
                      onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))} />
                    <Button variant="secondary" className="!px-3" onClick={() => setSeed(String(Math.floor(Math.random() * 99999)))}>
                      <Dice5 className="size-4" />
                    </Button>
                  </div>
                </Field>
              </div>
            )}
            <Button variant="primary" size="lg" className="w-full" icon={<Sparkles className="size-5" />}
              loading={running} disabled={!canUse} onClick={generate}>
              {t("generateVariants")}
            </Button>
            <Button variant="ghost" className="w-full" icon={<ImageUp className="size-4" />} disabled={!canUse || running}
              onClick={() => fileInput.current?.click()}>
              {t("uploadEdit")}
            </Button>
            <input ref={fileInput} type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && uploadEdit(e.target.files[0])} />
          </Card>
          {job && <JobPanel job={job} />}
        </aside>
      </div>

      <div>
        <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-white">
          {t("candidates")} <Badge>{num(candidates.length)}</Badge>
        </div>
        {candidates.length === 0 ? (
          <EmptyState icon={<Sparkles className="size-6" />} text={t("noCandidates")} />
        ) : (
          <div className="flex gap-3 overflow-x-auto pb-2">
            {candidates.map((c) => {
              const chosen = p.keyframe.chosen === c.id;
              return (
                <div key={c.id} className={clsx("group relative w-48 shrink-0 overflow-hidden rounded-2xl ring-2 transition animate-pop",
                  focus === c.id ? "ring-violet-400" : chosen ? "ring-emerald-400/70" : "ring-white/10 hover:ring-white/25")}>
                  <button className="relative block w-full" onClick={() => setFocus(focus === c.id ? null : c.id)}>
                    <img src={urls.candidate(p.id, c.id)} className="aspect-video w-full object-cover" />
                    {c.match !== undefined && c.match < 0.6 && (
                      <span title={t("poseChangedHint")}
                        className="absolute top-2 start-2 flex items-center gap-1 rounded-full bg-amber-500/90 px-2 py-0.5 text-[10px] font-semibold text-black">
                        <AlertTriangle className="size-3" /> {t("poseChanged")}
                      </span>
                    )}
                  </button>
                  <div className="flex items-center justify-between gap-1 bg-ink-850 p-2">
                    {chosen ? (
                      <span className="flex items-center gap-1 text-xs font-semibold text-emerald-300"><CheckCircle2 className="size-4" /> {t("chosen")}</span>
                    ) : (
                      <Button size="sm" variant="primary" onClick={() => choose(c)}>{t("choose")}</Button>
                    )}
                    <span className="text-[11px] tabular-nums text-zinc-500">#{num(c.frame)}</span>
                    <button onClick={() => api.deleteCandidate(p.id, c.id).then(setProject)}
                      className="rounded-lg p-1 text-zinc-500 hover:bg-rose-500/10 hover:text-rose-300">
                      <Trash2 className="size-3.5" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
      <StepFooter next="generate" nextDisabled={!p.keyframe.chosen || !p.mask_layers.ready} />
    </div>
  );
}
