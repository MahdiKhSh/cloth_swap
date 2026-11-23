import clsx from "clsx";
import { AlertTriangle, CheckCircle2, ChevronDown, Loader2, Square } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type Job } from "../api";
import { jobMessage, useT, type Key } from "../i18n";
import { Button, ProgressBar } from "../ui";

function useNow(active: boolean) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(id);
  }, [active]);
  return now;
}

const fmt = (s: number) => {
  s = Math.max(0, Math.round(s));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

export function JobPanel({ job, compact }: { job: Job; compact?: boolean }) {
  const t = useT();
  const [logs, setLogs] = useState(false);
  const running = job.status === "running" || job.status === "queued";
  const now = useNow(running);
  const elapsed = job.started ? (job.finished ?? now) - job.started : 0;
  const eta = running && job.progress > 0.08 ? (elapsed / job.progress) * (1 - job.progress) : null;

  return (
    <div className={clsx("rounded-2xl border p-4 animate-fade-in",
      job.status === "failed" ? "border-rose-500/30 bg-rose-500/[0.06]" : "border-white/10 bg-white/[0.03]")}>
      <div className="flex items-center gap-3">
        {running && <Loader2 className="size-5 shrink-0 animate-spin text-violet-300" />}
        {job.status === "done" && <CheckCircle2 className="size-5 shrink-0 text-emerald-400" />}
        {job.status === "failed" && <AlertTriangle className="size-5 shrink-0 text-rose-400" />}
        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="font-semibold text-white">{t(`job_${job.kind}` as Key)}</span>
            <span className="tabular-nums text-zinc-400">
              {job.status === "queued" ? t("queued") : job.status === "failed" ? t("failed")
                : job.status === "cancelled" ? t("cancelled") : `${Math.round(job.progress * 100)}%`}
            </span>
          </div>
          {!compact && <div className="mt-0.5 truncate text-xs text-zinc-400">{jobMessage(t, job.message)}</div>}
        </div>
        {running && (
          <Button size="sm" variant="ghost" icon={<Square className="size-3.5" />} onClick={() => api.cancel(job.id)}>
            {t("cancel")}
          </Button>
        )}
      </div>
      {running && (
        <div className="mt-3">
          <ProgressBar value={job.progress} active />
          <div className="mt-1.5 flex justify-between text-[11px] tabular-nums text-zinc-500">
            <span>{fmt(elapsed)}</span>
            {eta !== null && <span>~{fmt(eta)} {t("remaining")}</span>}
          </div>
        </div>
      )}
      {job.error && <p className="mt-3 whitespace-pre-wrap break-words text-xs leading-6 text-rose-200" dir="ltr">{job.error}</p>}
      {!compact && job.logs && job.logs.length > 0 && (
        <div className="mt-2">
          <button className="flex items-center gap-1 text-xs text-zinc-500 hover:text-zinc-300" onClick={() => setLogs(!logs)}>
            <ChevronDown className={clsx("size-3.5 transition", logs && "rotate-180")} /> {t("showLog")}
          </button>
          {logs && (
            <pre dir="ltr" className="mt-2 max-h-48 overflow-auto rounded-lg bg-black/40 p-3 text-[11px] leading-5 text-zinc-400">
              {job.logs.join("\n")}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}
