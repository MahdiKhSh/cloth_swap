import { Download, RefreshCw, SlidersHorizontal } from "lucide-react";
import { useState } from "react";
import { api, urls } from "../api";
import { CompareVideo } from "../components/Compare";
import { JobPanel } from "../components/JobPanel";
import { useAction, useJob } from "../hooks";
import { useLang, useNum, useT } from "../i18n";
import { useWs } from "../pages/Workspace";
import { Button, Card, SectionTitle, Segmented, Slider } from "../ui";

export function StepReview() {
  const t = useT();
  const fa = useLang((s) => s.lang) === "fa";
  const num = useNum();
  const { project: p, setProject, refresh, go } = useWs();
  const v = p.video!;
  const [comp, setComp] = useState(p.composite);
  const [jobId, setJobId] = useState<string | null>(null);
  const action = useAction();
  const job = useJob(jobId, refresh, () => setJobId(null));
  const running = !!jobId && (!job || job.status === "running" || job.status === "queued");
  const changed = JSON.stringify(comp) !== JSON.stringify(p.composite);
  const needsRecomposite = p.result?.stale && p.result.stale_reason === "composite";

  const apply = async () => {
    const saved = await action.run(() => api.patch(p.id, { composite: comp }));
    if (!saved) return;
    setProject(saved);
    const j = await action.run(() => api.composite(p.id));
    if (j) setJobId(j.id);
  };

  return (
    <div className="flex h-full flex-col gap-4">
      <SectionTitle title={t("reviewTitle")} hint={t("reviewHint")}
        right={
          <a href={urls.download(p.id)} download>
            <Button variant="primary" size="lg" icon={<Download className="size-5" />}>{t("download")}</Button>
          </a>
        } />
      {p.result?.stale && <div role="alert" className="rounded-lg border border-amber-400/30 bg-amber-400/10 px-4 py-2 text-sm text-amber-200">
        {p.result.stale_reason === "composite"
          ? (fa ? "تنظیمات ترکیب تغییر کرده است؛ برای دیدن نتیجه جدید، روی اعمال کلیک کن." : "Compositing settings changed. Apply them to refresh the result.")
          : (fa ? "این خروجی متعلق به لباس، ماسک یا تنظیمات قبلی است. ابتدا ویدئو را دوباره تولید کن." : "This result uses an earlier garment, mask or generation setup. Generate the video again.")}
      </div>}
      <div className="flex min-h-0 flex-1 flex-col gap-5 xl:flex-row">
        <div className="min-w-0 flex-1">
          <CompareVideo key={p.result?.updated} before={urls.source(p.id)} after={urls.result(p.id, p.result?.updated ?? 0)}
            offset={p.trim!.start / v.fps!} width={v.width!} height={v.height!} labels={[t("before"), t("after")]} />
        </div>
        <aside className="flex w-full shrink-0 flex-col gap-4 overflow-y-auto xl:w-80">
          <Card className="p-2">
            <div className="flex items-center gap-2 px-3 pt-2 text-sm font-semibold text-white">
              <SlidersHorizontal className="size-4 text-violet-300" /> {t("finishing")}
            </div>
            <div className="p-3">
              <Segmented value={comp.mode} onChange={(mode) => setComp({ ...comp, mode })}
                options={[{ value: "actor", label: t("modeActor") }, { value: "full", label: t("modeFull") }]} />
            </div>
            <Slider label={t("edgeSoftness")} value={comp.feather} min={0.002} max={0.04} step={0.002}
              onChange={(feather) => setComp({ ...comp, feather })} format={(x) => num(Math.round(x * 1000))} />
            <Slider label={t("maskGrow")} value={comp.dilate} min={0} max={0.06} step={0.004}
              onChange={(dilate) => setComp({ ...comp, dilate })} format={(x) => num(Math.round(x * 1000))} />
            <div className="p-3">
              <Button variant={changed ? "primary" : "secondary"} className="w-full" icon={<RefreshCw className="size-4" />}
                loading={running} disabled={(!changed && !needsRecomposite) || running || (!!p.result?.stale && p.result.stale_reason !== "composite")} onClick={apply}>
                {t("apply")}
              </Button>
            </div>
          </Card>
          {job && <JobPanel job={job} compact />}
          {p.result?.size && (
            <div className="px-1 text-xs text-zinc-500" dir="ltr">
              {fa ? "وضوح تولید مدل" : "Model render"}: {p.result.size[0]}×{p.result.size[1]} → {v.width}×{v.height}
            </div>
          )}
          <Button variant="ghost" onClick={() => go("generate")}>{t("regenerate")}</Button>
        </aside>
      </div>
    </div>
  );
}
