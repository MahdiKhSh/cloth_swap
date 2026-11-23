import clsx from "clsx";
import { ArrowLeft, ArrowRight, Check, Clapperboard, Film, Lock, Shirt, Sparkles, UserRound, Wand2 } from "lucide-react";
import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, type Job, type Project } from "../api";
import { navigate } from "../App";
import { GpuChip, LangButton, Logo } from "../components/Brand";
import { useLang, useT, type Key } from "../i18n";
import { StepActor } from "../steps/StepActor";
import { StepGarment } from "../steps/StepGarment";
import { StepGenerate } from "../steps/StepGenerate";
import { StepKeyframe } from "../steps/StepKeyframe";
import { StepReview } from "../steps/StepReview";
import { StepVideo } from "../steps/StepVideo";
import { Button, inputClass, useToasts } from "../ui";

export type StepId = "video" | "actor" | "garment" | "keyframe" | "generate" | "review";

const STEPS: { id: StepId; icon: typeof Film }[] = [
  { id: "video", icon: Film },
  { id: "actor", icon: UserRound },
  { id: "garment", icon: Shirt },
  { id: "keyframe", icon: Wand2 },
  { id: "generate", icon: Sparkles },
  { id: "review", icon: Clapperboard },
];

function stepState(p: Project, id: StepId): { available: boolean; done: boolean } {
  const hasVideo = !!p.video?.num_frames;
  const hasGarment = p.garment.images.length > 0 || !!p.garment.description.trim();
  switch (id) {
    case "video": return { available: hasVideo, done: hasVideo };
    case "actor": return { available: hasVideo, done: p.actor.tracked };
    case "garment": return { available: hasVideo, done: hasGarment };
    case "keyframe": return { available: p.actor.tracked && hasGarment, done: !!p.keyframe.chosen };
    case "generate": return { available: !!p.keyframe.chosen, done: !!p.result && !p.result.stale };
    case "review": return { available: !!p.result, done: !!p.result && !p.result.stale };
  }
}

interface Ctx {
  project: Project;
  setProject: (p: Project) => void;
  refresh: () => Promise<void>;
  go: (step: StepId) => void;
  job: (kind: string) => Job | undefined;
}
const WsContext = createContext<Ctx | null>(null);
export const useWs = () => useContext(WsContext)!;

export function StepFooter({ next, nextDisabled, children, beforeNext }: {
  next?: StepId; nextDisabled?: boolean; children?: ReactNode; beforeNext?: () => Promise<unknown>;
}) {
  const t = useT();
  const push = useToasts((s) => s.push);
  const { go: goTo } = useWs();
  const go = async (s: StepId) => {
    try {
      if (beforeNext) await beforeNext();
    } catch (error) {
      push(error instanceof Error ? error.message : String(error), "error");
      return;
    }
    goTo(s);
  };
  const rtl = useLang((s) => s.lang) === "fa";
  return (
    <div className="flex items-center justify-between gap-3 border-t border-white/5 pt-4">
      <div className="flex items-center gap-2">{children}</div>
      {next && (
        <Button variant="primary" size="lg" disabled={nextDisabled} onClick={() => go(next)}>
          {t("next")} {rtl ? <ArrowLeft className="size-4" /> : <ArrowRight className="size-4" />}
        </Button>
      )}
    </div>
  );
}

export function Workspace({ id, step }: { id: string; step?: StepId }) {
  const t = useT();
  const push = useToasts((s) => s.push);
  const [project, setProject] = useState<Project | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setProject(await api.project(id));
    } catch (e) {
      setError((e as Error).message);
    }
  }, [id]);
  // re-read on every step change: a step unlocked by a job that finished elsewhere must open
  const [checked, setChecked] = useState<string | undefined>(undefined);
  useEffect(() => { refresh().then(() => setChecked(step)); }, [refresh, step]);

  const running = project?.jobs?.some((j) => j.status === "running" || j.status === "queued");
  useEffect(() => {
    const timer = setInterval(refresh, running ? 1500 : 10000);
    return () => clearInterval(timer);
  }, [running, refresh]);

  if (error) return <div className="p-10 text-rose-300">{error}</div>;
  const spinner = <div className="flex h-full items-center justify-center"><div className="size-10 animate-spin rounded-full border-2 border-violet-400 border-t-transparent" /></div>;
  if (!project) return spinner;
  // a requested step that looks locked may only be stale: wait for the refresh before falling back
  if (step && !stepState(project, step).available && checked !== step) return spinner;

  const current: StepId = step && stepState(project, step).available ? step : "video";
  const go = (s: StepId) => navigate(id, s);
  const job = (kind: string) => [...(project.jobs ?? [])].reverse().find((j) => j.kind === kind);
  const ctx: Ctx = { project, setProject, refresh, go, job };

  return (
    <WsContext.Provider value={ctx}>
      <div className="flex h-full flex-col">
        <header className="flex h-16 shrink-0 items-center justify-between gap-4 border-b border-white/5 px-5">
          <div className="flex min-w-0 items-center gap-3">
            <button onClick={() => navigate()} className="rounded-xl p-1 transition hover:bg-white/[0.06]" title={t("projects")}><Logo small /></button>
            <span className="text-zinc-600">/</span>
            <input
              defaultValue={project.name}
              onBlur={async (e) => {
                if (e.target.value === project.name) return;
                try { setProject(await api.patch(id, { name: e.target.value })); }
                catch (error) {
                  e.target.value = project.name;
                  push(error instanceof Error ? error.message : String(error), "error");
                }
              }}
              className={clsx(inputClass, "h-9 w-36 border-transparent bg-transparent font-semibold hover:border-white/10 sm:w-64")}
            />
          </div>
          <div className="flex items-center gap-2"><GpuChip /><LangButton /></div>
        </header>

        <div className="flex min-h-0 flex-1">
          <nav className="w-16 shrink-0 overflow-y-auto border-e border-white/5 p-2 md:w-56 md:p-4 lg:w-72">
            <ol className="relative space-y-1.5">
              {STEPS.map((s, i) => {
                const st = stepState(project, s.id);
                const active = s.id === current;
                const Icon = s.icon;
                const runningHere = (project.jobs ?? []).some((j) =>
                  (j.status === "running" || j.status === "queued") &&
                  ({ track: "actor", keyframe: "keyframe", generate: "generate", composite: "review" } as Record<string, string>)[j.kind] === s.id);
                return (
                  <li key={s.id}>
                    <button
                      disabled={!st.available}
                      title={!st.available ? t("locked") : undefined}
                      onClick={() => go(s.id)}
                      className={clsx("group flex w-full items-center gap-3 rounded-2xl p-3 text-start transition",
                        active ? "bg-white/[0.08] ring-1 ring-white/10" : "hover:bg-white/[0.04]",
                        !st.available && "cursor-not-allowed opacity-40")}
                    >
                      <span className={clsx("relative flex size-10 shrink-0 items-center justify-center rounded-xl transition",
                        active ? "brand-gradient text-white shadow-lg shadow-fuchsia-500/25"
                          : st.done ? "bg-emerald-500/15 text-emerald-300" : "bg-white/[0.06] text-zinc-400")}>
                        {st.done && !active ? <Check className="size-5" /> : st.available ? <Icon className="size-5" /> : <Lock className="size-4" />}
                        {runningHere && <span className="absolute -top-1 -end-1 size-3 animate-pulse rounded-full bg-amber-400 ring-2 ring-ink-950" />}
                      </span>
                      <span className="hidden min-w-0 md:block">
                        <span className="flex items-center gap-2 text-sm font-semibold text-white">
                          <span className="text-xs tabular-nums text-zinc-500">{i + 1}</span>
                          {t(`step_${s.id}` as Key)}
                        </span>
                        <span className="block truncate text-xs text-zinc-500">{t(`step_${s.id}_d` as Key)}</span>
                      </span>
                    </button>
                  </li>
                );
              })}
            </ol>
          </nav>

          <main className="min-w-0 flex-1 overflow-hidden p-3 md:p-6" key={current}>
            {current === "video" && <StepVideo />}
            {current === "actor" && <StepActor />}
            {current === "garment" && <StepGarment />}
            {current === "keyframe" && <StepKeyframe />}
            {current === "generate" && <StepGenerate />}
            {current === "review" && <StepReview />}
          </main>
        </div>
      </div>
    </WsContext.Provider>
  );
}
