import clsx from "clsx";
import { ImagePlus, Info, X } from "lucide-react";
import { useRef, useState } from "react";
import { api, urls } from "../api";
import { useAction } from "../hooks";
import { useLang, useT } from "../i18n";
import { StepFooter, useWs } from "../pages/Workspace";
import { GarmentBiblePanel } from "../components/GarmentBiblePanel";
import { Card, Field, inputClass, SectionTitle } from "../ui";

const EXAMPLES = {
  fa: ["a black leather biker jacket over a white t-shirt", "a navy blue tailored suit with a white shirt", "a red hooded sweatshirt", "a beige trench coat"],
  en: ["a black leather biker jacket over a white t-shirt", "a navy blue tailored suit with a white shirt", "a red hooded sweatshirt", "a beige trench coat"],
};

export function StepGarment() {
  const t = useT();
  const lang = useLang((s) => s.lang);
  const { project: p, setProject } = useWs();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const upload = useAction();
  const edit = useAction();
  const [desc, setDesc] = useState(p.garment.description);
  const [subject, setSubject] = useState(p.prompt.subject);
  const [scene, setScene] = useState(p.prompt.scene);

  const add = async (files: FileList | File[]) => {
    if (p.garment.bible_id) return;
    for (const f of Array.from(files).slice(0, 3)) {
      const res = await upload.run(() => api.addGarment(p.id, f));
      if (res) setProject(res);
    }
  };
  const saveDesc = async (value: string) => {
    if (p.garment.bible_id || value === p.garment.description) return;
    const updated = await edit.run(() => api.patch(p.id, { garment: { description: value } }));
    if (updated) setProject(updated);
  };
  const savePrompt = async (values: Record<string, string>) => {
    const updated = await edit.run(() => api.patch(p.id, { prompt: values }));
    if (updated) setProject(updated);
  };

  return (
    <div className="flex h-full min-h-0 flex-col gap-4 overflow-y-auto pe-2">
      <SectionTitle title={t("garmentTitle")} hint={t("garmentHint")} />
      <GarmentBiblePanel project={p} currentDescription={desc} onApply={(updated) => { setProject(updated); setDesc(updated.garment.description); }} />
      <div className="grid shrink-0 grid-cols-1 gap-5 lg:grid-cols-2">
        <Card className="p-5">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            {p.garment.images.map((name) => (
              <div key={name} className="group relative aspect-[3/4] overflow-hidden rounded-2xl checker ring-1 ring-white/10 animate-pop">
                <img src={urls.garment(p.id, name)} className="h-full w-full object-contain" />
                <button disabled={!!p.garment.bible_id} onClick={async () => { const updated = await edit.run(() => api.removeGarment(p.id, name)); if (updated) setProject(updated); }}
                  className="absolute top-2 end-2 rounded-full bg-black/70 p-1.5 text-white opacity-0 transition group-hover:opacity-100 hover:bg-rose-500">
                  <X className="size-4" />
                </button>
              </div>
            ))}
            {!p.garment.bible_id && <button
              onClick={() => input.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setOver(true); }}
              onDragLeave={() => setOver(false)}
              onDrop={(e) => { e.preventDefault(); setOver(false); add(e.dataTransfer.files); }}
              className={clsx("flex aspect-[3/4] flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed text-sm transition",
                over ? "border-violet-400 bg-violet-500/10 text-violet-200" : "border-white/15 text-zinc-400 hover:border-violet-400/50 hover:text-zinc-200")}>
              {upload.busy ? <span className="size-6 animate-spin rounded-full border-2 border-violet-300 border-t-transparent" /> : <ImagePlus className="size-7" />}
              {t("addGarmentPhoto")}
            </button>}
            <input ref={input} type="file" accept="image/*" multiple hidden onChange={(e) => e.target.files && add(e.target.files)} />
          </div>
        </Card>

        <Card className="space-y-5 p-5">
          <Field label={t("garmentDesc")}>
            <textarea className={clsx(inputClass, "min-h-24 resize-y")} dir="auto" value={desc} placeholder={t("garmentDescPh")}
              disabled={!!p.garment.bible_id} onChange={(e) => setDesc(e.target.value)} onBlur={() => saveDesc(desc)} />
          </Field>
          <div>
            <div className="mb-2 text-xs text-zinc-500">{t("examples")}</div>
            <div className="flex flex-wrap gap-2">
              {!p.garment.bible_id && EXAMPLES[lang].map((ex) => (
                <button key={ex} dir="ltr" onClick={() => { setDesc(ex); saveDesc(ex); }}
                  className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs text-zinc-300 transition hover:border-violet-400/40 hover:text-white">
                  {ex}
                </button>
              ))}
            </div>
          </div>
          <div className="h-px bg-white/5" />
          <Field label={t("subject")}>
            <input className={inputClass} dir="auto" value={subject} placeholder={t("subjectPh")}
              onChange={(e) => setSubject(e.target.value)} onBlur={() => savePrompt({ subject })} />
          </Field>
          <Field label={t("scene")}>
            <textarea className={clsx(inputClass, "min-h-20 resize-y")} dir="auto" value={scene} placeholder={t("scenePh")}
              onChange={(e) => setScene(e.target.value)} onBlur={() => savePrompt({ scene })} />
          </Field>
          <p className="flex items-center gap-2 text-xs text-zinc-500"><Info className="size-3.5" /> {t("promptTip")}</p>
        </Card>
      </div>
      <StepFooter next="keyframe" nextDisabled={!p.actor.tracked || (!p.garment.images.length && !desc.trim())}
        beforeNext={() => api.patch(p.id, p.garment.bible_id ? { prompt: { subject, scene } } : { garment: { description: desc }, prompt: { subject, scene } }).then(setProject)} />
    </div>
  );
}
