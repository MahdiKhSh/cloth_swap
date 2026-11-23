import clsx from "clsx";
import { ChevronDown, Gauge, Rocket, Sparkles, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { api, urls, type SystemInfo } from "../api";
import { JobPanel } from "../components/JobPanel";
import { useAction, useJob } from "../hooks";
import { useLang, useT, type Key } from "../i18n";
import { StepFooter, useWs } from "../pages/Workspace";
import { Button, Card, Field, inputClass, SectionTitle, Toggle } from "../ui";

const PRESET_ICON = { preview: Zap, standard: Gauge, high: Rocket } as const;
export function StepGenerate() {
  const t = useT();
  const fa = useLang((s) => s.lang) === "fa";
  const { project: p, setProject, refresh, go, job: lastJob } = useWs();
  const [sys, setSys] = useState<SystemInfo | null>(null);
  const [gen, setGen] = useState(p.generate);
  const [full, setFull] = useState(p.prompt.full ?? "");
  const [advanced, setAdvanced] = useState(false);
  const [jobId, setJobId] = useState<string | null>(() => {
    const j = lastJob("generate");
    return j && (j.status === "running" || j.status === "queued") ? j.id : null;
  });
  const action = useAction();
  useEffect(() => {
    let stopped = false;
    const update = () => api.system().then((s) => { if (!stopped) setSys(s); }).catch(() => {});
    update(); const timer = setInterval(update, 15000);
    return () => { stopped = true; clearInterval(timer); };
  }, []);

  const job = useJob(jobId, async () => { await refresh(); go("review"); }, () => setJobId(null));
  const running = !!jobId && (!job || job.status === "running" || job.status === "queued");

  const presets = Object.keys(sys?.presets ?? { preview: 0, standard: 0, high: 0 }) as (keyof typeof PRESET_ICON)[];
  const activeArea = sys?.presets[gen.preset]?.target_area;
  const workSize = p.video?.width && p.video?.height && activeArea ? (() => {
    const aspect = p.video.width! / p.video.height!;
    return [Math.max(32, Math.round(Math.sqrt(activeArea * aspect) / 32) * 32),
            Math.max(32, Math.round(Math.sqrt(activeArea / aspect) / 32) * 32)];
  })() : null;
  const builtPrompt = `${p.prompt.subject || "A person"}, wearing ${p.garment.description || "the outfit shown in the reference image"}.${p.prompt.scene ? " " + p.prompt.scene : ""}`;

  const start = async () => {
    const saved = await action.run(() => api.patch(p.id, { generate: { ...gen, output: "original", upscale: "resize" }, prompt: { full } }));
    if (!saved) return;
    setProject(saved);
    const j = await action.run(() => api.generate(p.id));
    if (j) setJobId(j.id);
  };

  return (
    <div className="flex h-full flex-col gap-4">
      <SectionTitle title={t("generateTitle")} hint={t("generateHint")} />
      <div className="rounded-xl border border-amber-500/25 bg-amber-500/5 px-4 py-3 text-xs leading-6 text-amber-200">
        {workSize ? (fa ? `جزئیات تولیدی مدل: ${workSize[0]}×${workSize[1]} | اندازه فایل نهایی: ${p.video?.width}×${p.video?.height}. ` :
          `Generated detail: ${workSize[0]}×${workSize[1]} | Export container: ${p.video?.width}×${p.video?.height}. `) : ""}
        {fa ? "تطبیق ابعاد با ویدئوی اصلی به معنای تولید واقعی جزئیات 4K نیست؛ از AI Upscaler استفاده نمی‌شود." :
          "Matching the original video size does not create native 4K detail. No AI upscaler is used."}
      </div>
      <div className="grid min-h-0 flex-1 grid-cols-1 gap-5 overflow-y-auto xl:grid-cols-[1fr_380px]">
        <div className="space-y-5">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {presets.map((id) => {
              const Icon = PRESET_ICON[id] ?? Sparkles;
              const selected = gen.preset === id;
              return (
                <button key={id} disabled={running} onClick={() => setGen({ ...gen, preset: id, steps: null })}
                  className={clsx("glass rounded-2xl p-5 text-start transition-all",
                    selected ? "ring-2 ring-violet-400 shadow-xl shadow-violet-900/30" : "hover:-translate-y-0.5 hover:ring-1 hover:ring-white/20")}>
                  <div className={clsx("mb-4 flex size-11 items-center justify-center rounded-xl",
                    selected ? "brand-gradient text-white" : "bg-white/[0.07] text-zinc-300")}>
                    <Icon className="size-5" />
                  </div>
                  <div className="font-bold text-white">{t(`preset_${id}` as Key)}</div>
                  <div className="mt-1 text-xs text-zinc-400">{t(`preset_${id}_d` as Key)}</div>

                </button>
              );
            })}
          </div>

          <p className="text-sm leading-relaxed text-zinc-400">{t("outputResD")}</p>
          {p.garment.bible_id && <p className="rounded-lg border border-violet-400/20 bg-violet-400/5 p-3 text-xs text-violet-200">{fa ? "لباس مرجع به این سکانس متصل است؛ برای ثبات لباس در سکانس‌های دیگر، همان لباس را از کتابخانه انتخاب کنید." : "A garment bible is linked. Reuse that saved outfit in other shots to maintain continuity."}</p>}

          <Card className="p-4">
            {sys?.relight_available && (
              <Toggle checked={gen.relight} onChange={(v) => setGen({ ...gen, relight: v })} label={t("relight")} hint={t("relightD")} />
            )}
            <button className="mt-1 flex items-center gap-1 px-3 text-xs text-zinc-500 hover:text-zinc-300" onClick={() => setAdvanced(!advanced)}>
              <ChevronDown className={clsx("size-3.5 transition", advanced && "rotate-180")} /> {t("advanced")}
            </button>
            {advanced && (
              <div className="grid grid-cols-1 gap-4 p-3 md:grid-cols-2 animate-fade-in">
                <Field label={t("seed")}>
                  <input dir="ltr" className={inputClass} value={gen.seed} inputMode="numeric"
                    onChange={(e) => setGen({ ...gen, seed: Number(e.target.value.replace(/\D/g, "")) || 0 })} />
                </Field>
                <Field label={t("steps")}>
                  <input dir="ltr" className={inputClass} value={gen.steps ?? sys?.presets[gen.preset]?.steps ?? 8} inputMode="numeric"
                    onChange={(e) => setGen({ ...gen, steps: Number(e.target.value.replace(/\D/g, "")) || null })} />
                </Field>
              </div>
            )}
          </Card>

          <Card className="p-4">
            <Field label={t("fullPrompt")}>
              <textarea dir="ltr" className={clsx(inputClass, "min-h-24 text-xs leading-5")} value={full} placeholder={builtPrompt}
                onChange={(e) => setFull(e.target.value)} />
            </Field>
          </Card>

          {job && <JobPanel job={job} />}
        </div>

        <Card className="flex flex-col gap-4 self-start p-4">
          {p.keyframe.chosen && (
            <img src={urls.candidate(p.id, p.keyframe.chosen)} className="w-full rounded-xl ring-1 ring-white/10" />
          )}
          {sys?.model_ready === false && <p className="text-sm text-amber-300">{t("animateNotReady")}</p>}
          <Button variant="primary" size="lg" icon={<Sparkles className="size-5" />} loading={running} disabled={sys?.model_ready === false} onClick={start}>
            {p.result ? t("regenerate") : t("start")}
          </Button>
        </Card>
      </div>
      <StepFooter next="review" nextDisabled={!p.result} />
    </div>
  );
}
