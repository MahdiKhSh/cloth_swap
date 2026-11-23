import { BookOpen, Camera, Check, CopyCheck, RefreshCw, Save, Shirt, Sparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { api, urls, type GarmentBible, type Project } from "../api";
import { useLang } from "../i18n";
import { Card, inputClass } from "../ui";

const KEYS = ["description", "type", "color", "fabric", "pattern", "fit", "collar", "sleeves", "closures", "pockets", "logo", "seams", "notes"] as const;
type DetailField = typeof KEYS[number];
const LABELS: Record<DetailField, [string, string]> = {
  description: ["شرح کلی لباس", "Garment description"],
  type: ["نوع لباس", "Garment type"],
  color: ["رنگ دقیق", "Exact color"],
  fabric: ["جنس پارچه", "Fabric"],
  pattern: ["طرح و بافت", "Pattern / texture"],
  fit: ["برش و اندازه", "Cut / fit"],
  collar: ["یقه", "Collar / neckline"],
  sleeves: ["آستین", "Sleeves"],
  closures: ["دکمه و زیپ", "Buttons / zippers"],
  pockets: ["جیب‌ها", "Pockets"],
  logo: ["لوگو و نشان", "Logo / branding"],
  seams: ["دوخت", "Seams / stitching"],
  notes: ["جزئیات ثابت دیگر", "Other fixed details"],
};
const VIEWS = ["front", "back", "side", "detail"] as const;
const VIEW_LABELS: Record<(typeof VIEWS)[number], [string, string]> = {
  front: ["جلو", "Front"], back: ["پشت", "Back"], side: ["بغل", "Side"], detail: ["جزئیات", "Detail"],
};

export function GarmentBiblePanel({ project, currentDescription, onApply }: {
  project: Project;
  currentDescription: string;
  onApply: (value: Project) => void;
}) {
  const fa = useLang((s) => s.lang) === "fa";
  const tr = (f: string, e: string) => fa ? f : e;
  const [library, setLibrary] = useState<GarmentBible[]>([]);
  const [selected, setSelected] = useState(project.garment.bible_id || "");
  const [draft, setDraft] = useState<Partial<GarmentBible>>({});
  const [name, setName] = useState("");
  const [view, setView] = useState<(typeof VIEWS)[number]>("front");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const bible = library.find((b) => b.id === selected);
  const active = project.garment.bible_id === selected && !!selected;
  const outdated = active && !!bible && project.garment.bible_revision !== bible.revision;

  const load = async () => setLibrary(await api.garmentBibles());
  useEffect(() => { load().catch((e) => setError(String(e))); }, []);
  useEffect(() => {
    if (project.garment.bible_id && !selected) setSelected(project.garment.bible_id);
  }, [project.garment.bible_id, selected]);
  useEffect(() => {
    const next = library.find((b) => b.id === selected);
    if (next) {
      const fields: Partial<GarmentBible> = { name: next.name };
      for (const key of KEYS) fields[key] = next[key];
      setDraft(fields);
    } else setDraft({});
  }, [selected, library]);

  const perform = async (callback: () => Promise<void>) => {
    if (busy) return;
    setBusy(true); setError(""); setMessage("");
    try { await callback(); }
    catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setBusy(false); }
  };

  const pendingChanges = (): Partial<GarmentBible> => {
    const changes: Partial<GarmentBible> = {};
    if (!bible) return changes;
    if (draft.name !== bible.name) changes.name = draft.name;
    for (const key of KEYS) if (draft[key] !== bible[key]) changes[key] = draft[key];
    return changes;
  };

  const create = () => perform(async () => {
    const created = await api.createGarmentBible({
      name: name.trim() || project.name + " · Outfit",
      description: currentDescription.trim(),
      from_project_id: project.id,
    });
    setSelected(created.id);
    const applied = await api.applyGarmentBible(project.id, created.id);
    onApply(applied);
    await load();
    setMessage(tr("لباس مرجع ساخته و به این سکانس متصل شد.", "Garment bible created and linked to this shot."));
  });

  const apply = () => perform(async () => {
    if (!selected) return;
    onApply(await api.applyGarmentBible(project.id, selected));
    setMessage(tr("این سکانس به لباس مرجع متصل شد. فریم مرجع قبلی برای جلوگیری از اختلاف لباس لغو شد.",
      "Outfit linked. The previous keyframe was invalidated to prevent mismatched designs."));
  });

  const save = () => perform(async () => {
    if (!selected || !bible) return;
    const changes = pendingChanges();
    if (Object.keys(changes).length) await api.updateGarmentBible(selected, changes, bible.revision);
    const synced = await api.syncGarmentBible(selected);
    await load();
    if (project.garment.bible_id === selected && synced.synced.includes(project.id)) {
      onApply(await api.project(project.id));
    }
    setMessage(tr(`${synced.synced.length} سکانس همگام شد. ${synced.skipped_running.length} سکانس در حال پردازش بود.`,
      `${synced.synced.length} shots synced; ${synced.skipped_running.length} busy shots skipped.`));
  });

  const upload = (file: File | undefined) => perform(async () => {
    if (!selected || !file || !bible) return;
    // Saving a new reference picture refreshes the bible from the API.
    // Persist any unsaved form edits first instead of silently discarding them.
    const changes = pendingChanges();
    const updated = Object.keys(changes).length
      ? await api.updateGarmentBible(selected, changes, bible.revision) : bible;
    await api.uploadBibleImage(selected, view, file, updated.revision);
    const synced = await api.syncGarmentBible(selected);
    await load();
    if (project.garment.bible_id === selected && synced.synced.includes(project.id)) onApply(await api.project(project.id));
    setMessage(tr("تصویر مرجع ذخیره و سکانس‌ها همگام شدند؛ فریم کلیدی جدید انتخاب کن.",
      "Reference image updated and linked shots synced. Choose a new keyframe."));
  });

  return (
    <Card className="border border-violet-400/25 bg-violet-400/[0.035] p-5">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-base font-bold text-white"><BookOpen className="size-5 text-violet-300" /> {tr("کتابخانه لباس ثابت", "Garment Bible")}</div>
          <p className="mt-1 text-xs leading-6 text-zinc-400">{tr("یک لباس را تعریف کن و در تمام سکانس‌های همان بازیگر دوباره استفاده کن. جزئیات لباس با هر نسخه قفل می‌شود.",
            "Define an outfit once, reuse it in every scene, and version-lock its details.")}</p>
        </div>
        {active && <span className="flex items-center gap-1 rounded-full border border-emerald-400/30 px-3 py-1 text-xs text-emerald-300"><Check className="size-3" /> {tr("متصل به سکانس", "Linked to shot")}</span>}
      </div>
      <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
        <select className={inputClass} aria-label={tr("انتخاب لباس مرجع", "Choose a garment bible")} value={selected} onChange={(e) => setSelected(e.target.value)}>
          <option value="">{tr("یک لباس ذخیره‌شده انتخاب کن...", "Choose a saved outfit...")}</option>
          {library.map((b) => <option value={b.id} key={b.id}>{b.name} · {b.id.slice(-4)} (v{b.revision})</option>)}
        </select>
        <button disabled={!selected || busy || (active && !outdated)} onClick={apply}
          className="flex items-center justify-center gap-2 rounded-xl bg-violet-500 px-4 py-2 text-sm font-semibold text-white hover:bg-violet-400 disabled:cursor-not-allowed disabled:opacity-40">
          <CopyCheck className="size-4" /> {tr("اعمال روی این سکانس", "Apply to this shot")}
        </button>
      </div>
      {!bible && <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-white/10 pt-4">
        <input className={`${inputClass} max-w-80 flex-1`} placeholder={tr("نام لباس، مثلاً کت قرمز صحنه اول", "Outfit name, e.g. red jacket")}
          value={name} onChange={(e) => setName(e.target.value)} />
        <button onClick={create} disabled={busy || (!project.garment.images.length && !currentDescription.trim())}
          className="flex items-center gap-2 rounded-xl border border-white/20 px-4 py-2 text-sm text-white hover:bg-white/10 disabled:opacity-40">
          <Sparkles className="size-4" /> {tr("ساخت از لباس فعلی", "Create from current outfit")}
        </button>
        <p className="w-full text-xs text-zinc-500">{tr("اول تصویر یا توضیح لباس را در بخش پایین وارد کن، سپس یک لباس دائمی بساز.",
          "Add garment photos or a description below first, then save a reusable bible.")}</p>
      </div>}
      {bible && <div className="mt-5 space-y-4 border-t border-white/10 pt-4">
        <div className="flex flex-wrap gap-2">
          {bible.images.map((photo) => <div className="w-24" key={photo.view}>
            <img className="aspect-[3/4] w-full rounded-xl border border-white/10 bg-black/20 object-contain" src={urls.bibleImage(bible.id, photo.file, bible.revision)} alt={photo.view} />
            <p className="mt-1 text-center text-xs text-zinc-400">{VIEW_LABELS[photo.view][fa ? 0 : 1]}</p>
          </div>)}
          {!bible.images.length && <span className="text-xs text-amber-300">{tr("برای کیفیت بهتر چند نمای لباس اضافه کن.", "Add multiple views of the garment for better references.")}</span>}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Camera className="size-4 text-violet-300" />
          <select className={`${inputClass} max-w-36`} value={view} onChange={(e) => setView(e.target.value as typeof view)}>
            {VIEWS.map((v) => <option key={v} value={v}>{VIEW_LABELS[v][fa ? 0 : 1]}</option>)}
          </select>
          <input type="file" accept="image/png,image/jpeg,image/webp" aria-label={tr("تصویر مرجع لباس", "Garment reference photo")}
            className="min-w-0 max-w-72 text-xs text-zinc-300 file:me-3 file:rounded-lg file:border-0 file:bg-white/10 file:px-3 file:py-2 file:text-white"
            disabled={busy} onChange={(e) => { const f = e.target.files?.[0]; e.currentTarget.value = ""; if (f) upload(f); }} />
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          <label className="block text-xs text-zinc-400">{tr("نام لباس", "Outfit name")}
            <input className={`${inputClass} mt-1 w-full`} dir="auto" value={draft.name ?? ""}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
          </label>
          {KEYS.filter((key) => ["description", "color", "fabric", "collar", "sleeves"].includes(key)).map((key) => <label key={key} className={`block text-xs text-zinc-400 ${key === "description" ? "md:col-span-2" : ""}`}>
            {LABELS[key][fa ? 0 : 1]}
            {key === "description" ? <textarea rows={2} dir="auto" className={`${inputClass} mt-1 w-full`} value={draft[key] ?? ""}
              onChange={(e) => setDraft({ ...draft, [key]: e.target.value })} /> :
              <input dir="auto" className={`${inputClass} mt-1 w-full`} value={draft[key] ?? ""}
                onChange={(e) => setDraft({ ...draft, [key]: e.target.value })} />}
          </label>)}
        </div>
        <details className="rounded-xl border border-white/10 p-3">
          <summary className="cursor-pointer text-sm font-medium text-violet-200">{tr("جزئیات تخصصی لباس (طرح، برش، جیب، دوخت، لوگو...)", "Advanced garment details (pattern, fit, pockets, seams, logo...)")}</summary>
          <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {KEYS.filter((key) => !["description", "color", "fabric", "collar", "sleeves"].includes(key)).map((key) => <label className={`block text-xs text-zinc-400 ${key === "notes" ? "md:col-span-2" : ""}`} key={key}>
              {LABELS[key][fa ? 0 : 1]}
              {key === "notes" ? <textarea dir="auto" rows={2} className={`${inputClass} mt-1 w-full`} value={draft[key] ?? ""}
                onChange={(e) => setDraft({ ...draft, [key]: e.target.value })} /> :
                <input dir="auto" className={`${inputClass} mt-1 w-full`} value={draft[key] ?? ""}
                  onChange={(e) => setDraft({ ...draft, [key]: e.target.value })} />}
            </label>)}
          </div>
        </details>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-xs text-zinc-500">{tr("نسخه", "Revision")} {bible.revision} · {tr("هر تغییر، فریم‌های مرجع قبلی را نامعتبر می‌کند.", "Changes invalidate old keyframes.")}</p>
          <button disabled={busy} onClick={save} className="flex items-center gap-2 rounded-xl bg-violet-500 px-4 py-2 text-sm font-semibold text-white hover:bg-violet-400 disabled:opacity-50">
            {busy ? <RefreshCw className="size-4 animate-spin" /> : <Save className="size-4" />}
            {tr("ذخیره و همگام‌سازی همه سکانس‌ها", "Save & sync all linked shots")}
          </button>
        </div>
        {outdated && <p className="text-xs text-amber-300">{tr("نسخه لباس این سکانس قدیمی است؛ نسخه جدید را اعمال کن.", "This shot uses an old garment revision. Apply the new one.")}</p>}
        {!active && <p className="flex items-center gap-2 text-xs text-zinc-400"><Shirt className="size-4" /> {tr("برای استفاده از این لباس، دکمه «اعمال روی این سکانس» را بزن.", "Use Apply to link this outfit to this shot.")}</p>}
      </div>}
      {error && <p role="alert" className="mt-3 rounded-lg bg-rose-500/10 p-3 text-xs text-rose-300">{error}</p>}
      {message && <p role="status" className="mt-3 rounded-lg bg-emerald-500/10 p-3 text-xs text-emerald-300">{message}</p>}
    </Card>
  );
}
