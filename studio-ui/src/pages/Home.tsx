import clsx from "clsx";
import { Clock, Film, Sparkles, Trash2, UploadCloud } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, createProject, urls, type Project } from "../api";
import { navigate } from "../App";
import { GpuChip, LangButton, Logo } from "../components/Brand";
import { useJob } from "../hooks";
import { useLang, useNum, useT } from "../i18n";
import { Card, ProgressBar, useToasts } from "../ui";

function progressOf(p: Project) {
  return [p.trim, p.actor.tracked, p.garment.images.length || p.garment.description, p.keyframe.chosen, p.result && !p.result.stale].filter(Boolean).length;
}

export function Home() {
  const t = useT();
  const num = useNum();
  const lang = useLang((s) => s.lang);
  const push = useToasts((s) => s.push);
  const [projects, setProjects] = useState<Project[] | null>(null);
  const [upload, setUpload] = useState<{ name: string; progress: number } | null>(null);
  const [importJob, setImportJob] = useState<{ job: string; project: string } | null>(null);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const load = () => api.projects().then(setProjects).catch((e) => push(e.message, "error"));
  useEffect(() => { load(); }, []);

  useJob(importJob?.job ?? null, () => navigate(importJob!.project, "video"),
    (j) => {
      if (j.status === "failed" || j.status === "cancelled") {
        if (j.status === "failed") push(j.error ?? t("failed"), "error");
        setUpload(null);
        setImportJob(null);
      }
    });

  const start = async (file: File) => {
    setUpload({ name: file.name, progress: 0 });
    try {
      const res = await createProject(file, file.name.replace(/\.[^.]+$/, ""), (f) => setUpload({ name: file.name, progress: f }));
      setImportJob({ job: res.job.id, project: res.project.id });
    } catch (e) {
      push((e as Error).message, "error");
      setUpload(null);
    }
  };

  const remove = async (p: Project) => {
    if (!confirm(t("confirmDelete"))) return;
    await api.remove(p.id).catch((e) => push(e.message, "error"));
    load();
  };

  const date = (s: number) =>
    new Intl.DateTimeFormat(lang === "fa" ? "fa-IR" : "en-US", { dateStyle: "medium", timeStyle: "short" }).format(s * 1000);

  return (
    <div className="mx-auto min-h-full max-w-7xl px-6 pb-16">
      <header className="flex items-center justify-between py-6">
        <Logo />
        <div className="flex items-center gap-2"><GpuChip /><LangButton /></div>
      </header>

      <section className="relative mt-6 overflow-hidden rounded-3xl border border-white/10 p-10 animate-fade-in">
        <div className="absolute inset-0 -z-10 brand-gradient opacity-[0.12]" />
        <div className="absolute -top-24 end-[-10%] -z-10 size-96 rounded-full bg-fuchsia-500/20 blur-3xl" />
        <div className="flex flex-wrap items-center justify-between gap-8">
          <div className="max-w-xl">
            <div className="mb-3 inline-flex items-center gap-2 rounded-full bg-white/10 px-3 py-1 text-xs text-violet-100">
              <Sparkles className="size-3.5" /> SAM3 · Qwen-Image-Edit · SCAIL-2
            </div>
            <h1 className="text-4xl font-black leading-tight tracking-tight"><span className="brand-text">{t("appName")}</span></h1>
            <p className="mt-3 text-base leading-8 text-zinc-300">{t("tagline")}</p>
          </div>

          <div
            onClick={() => !upload && input.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f && !upload) start(f); }}
            className={clsx("glass flex w-full max-w-md cursor-pointer flex-col items-center justify-center gap-3 rounded-3xl border-2 border-dashed p-8 text-center transition-all",
              over ? "scale-[1.02] border-violet-400 bg-violet-500/10" : "border-white/15 hover:border-violet-400/50 hover:bg-white/[0.05]")}>
            <input ref={input} type="file" accept="video/*" hidden onChange={(e) => e.target.files?.[0] && start(e.target.files[0])} />
            {upload ? (
              <div className="w-full">
                <Film className="mx-auto mb-3 size-10 text-violet-300" />
                <div className="mb-3 truncate text-sm font-medium text-white">{upload.name}</div>
                <ProgressBar value={importJob ? 1 : upload.progress} active />
                <div className="mt-2 text-xs text-zinc-400">
                  {importJob ? t("importing") : `${t("uploading")} ${Math.round(upload.progress * 100)}%`}
                </div>
              </div>
            ) : (
              <>
                <div className="brand-gradient flex size-14 items-center justify-center rounded-2xl shadow-xl shadow-fuchsia-500/30">
                  <UploadCloud className="size-7 text-white" />
                </div>
                <div className="text-base font-semibold text-white">{t("newProject")}</div>
                <div className="text-sm text-zinc-300">{t("dropVideo")}</div>
                <div className="text-xs text-zinc-500">{t("dropVideoHint")}</div>
              </>
            )}
          </div>
        </div>
      </section>

      <h2 className="mb-4 mt-12 text-lg font-bold text-white">{t("projects")}</h2>
      {projects && projects.length === 0 && (
        <p className="rounded-2xl border border-dashed border-white/10 p-10 text-center text-sm text-zinc-500">{t("noProjects")}</p>
      )}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {projects?.map((p, i) => (
          <Card key={p.id} className="group overflow-hidden transition hover:-translate-y-0.5 hover:border-violet-400/30 hover:shadow-2xl hover:shadow-violet-900/20">
            <button className="block w-full text-start animate-fade-in" style={{ animationDelay: `${i * 40}ms` }} onClick={() => navigate(p.id)}>
              <div className="relative aspect-video overflow-hidden bg-black">
                {p.video?.num_frames ? (
                  <img src={urls.thumb(p.id, p.keyframe.frame ?? Math.floor(p.video.num_frames / 2))}
                    className="h-full w-full object-cover transition duration-500 group-hover:scale-105" />
                ) : <div className="h-full w-full checker" />}
                <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent" />
                {p.result && !p.result.stale && (
                  <span className="absolute top-3 start-3 rounded-full bg-emerald-500/90 px-2 py-0.5 text-[11px] font-semibold text-white">{t("done")}</span>
                )}
                <div className="absolute inset-x-3 bottom-3 flex gap-1">
                  {Array.from({ length: 5 }, (_, k) => (
                    <div key={k} className={clsx("h-1 flex-1 rounded-full", k < progressOf(p) ? "brand-gradient" : "bg-white/25")} />
                  ))}
                </div>
              </div>
              <div className="p-4">
                <div className="truncate font-semibold text-white">{p.name || p.id}</div>
                <div className="mt-1 flex items-center gap-3 text-xs text-zinc-500">
                  <span className="flex items-center gap-1"><Clock className="size-3" /> {date(p.updated)}</span>
                  {p.video?.num_frames && <span>{num(p.video.num_frames)} {t("frames")}</span>}
                </div>
              </div>
            </button>
            <div className="flex justify-end border-t border-white/5 px-2 py-1.5 opacity-0 transition group-hover:opacity-100">
              <button title={t("deleteProject")} onClick={() => remove(p)} className="rounded-lg p-1.5 text-zinc-500 hover:bg-rose-500/10 hover:text-rose-300">
                <Trash2 className="size-4" />
              </button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
