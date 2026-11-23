import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type Job } from "./api";
import { useT } from "./i18n";
import { useToasts } from "./ui";

/** Polls a job until it finishes; calls onDone once when it succeeds. */
export function useJob(jobId: string | null, onDone?: (job: Job) => void, onEnd?: (job: Job) => void) {
  const [job, setJob] = useState<Job | null>(null);
  const done = useRef(onDone);
  const end = useRef(onEnd);
  done.current = onDone;
  end.current = onEnd;

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      return;
    }
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const j = await api.job(jobId);
        if (stop) return;
        setJob(j);
        if (j.status === "done") done.current?.(j);
        if (["done", "failed", "cancelled"].includes(j.status)) {
          end.current?.(j);
          return;
        }
      } catch {
        /* transient */
      }
      timer = setTimeout(tick, 900);
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [jobId]);
  return job;
}

/** Wraps an async action: busy flag + translated error toast. */
export function useAction() {
  const push = useToasts((s) => s.push);
  const t = useT();
  const [busy, setBusy] = useState(false);
  const run = useCallback(
    async <T,>(fn: () => Promise<T>): Promise<T | undefined> => {
      setBusy(true);
      try {
        return await fn();
      } catch (e) {
        const known = e instanceof ApiError && (e.message === "gpu_busy" || e.message === "english_only");
        const msg = known ? t(e.message as "gpu_busy" | "english_only") : (e as Error).message;
        push(msg, "error");
        return undefined;
      } finally {
        setBusy(false);
      }
    },
    [push, t],
  );
  return { busy, run };
}

export function useDebounced<T>(value: T, ms: number) {
  const [v, setV] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setV(value), ms);
    return () => clearTimeout(id);
  }, [value, ms]);
  return v;
}

export function useKey(handler: (e: KeyboardEvent) => void, deps: unknown[]) {
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") return;
      handler(e);
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
