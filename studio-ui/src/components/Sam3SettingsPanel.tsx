import { useEffect, useMemo, useState } from "react";
import { Settings2, RotateCcw, Save, X, AlertTriangle, SlidersHorizontal } from "lucide-react";
import { api, type Sam3Controls } from "../api";
import { Button, inputClass } from "../ui";

type Value = number | string | boolean;

// Values are stored PER PROJECT. Presets only change relevant controls, not model weights.
const presets: Record<string, Record<string, Value>> = {
  "دقیق‌تر (کندتر)": {
    part_tracking_max_side: 4096, part_mask_resample: "linear",
    part_mask_blur_px: 0, part_mask_dilate_px: 0, image_size: 1008,
    tracker_mask_threshold: 0, part_fill_holes: false,
  },
  "متعادل": {
    part_tracking_max_side: 1440, part_mask_resample: "linear", image_size: 1008,
    part_mask_blur_px: 0, part_mask_dilate_px: 0,
  },
  "سریع (دقت پایین‌تر)": {
    part_tracking_max_side: 960, part_mask_resample: "nearest", image_size: 1008,
    part_mask_blur_px: 0,
  },
};

export function Sam3SettingsPanel({ projectId, onUpdated, disabled = false }: {
  projectId: string;
  onUpdated: () => Promise<unknown>;
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<Sam3Controls | null>(null);
  const [values, setValues] = useState<Record<string, Value>>({});
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [advanced, setAdvanced] = useState(false);

  useEffect(() => {
    if (!open) return;
    let ignore = false;
    setLoading(true); setError("");
    api.sam3Settings(projectId).then(response => {
      if (!ignore) { setData(response); setValues(response.values); }
    }).catch(e => { if (!ignore) setError(String((e as Error).message ?? e)); })
      .finally(() => { if (!ignore) setLoading(false); });
    return () => { ignore = true; };
  }, [open, projectId]);

  const changed = useMemo(() => data ? Object.keys(values).filter(key => values[key] !== data.values[key]) : [], [data, values]);
  const update = (key: string, value: Value) => setValues(v => ({...v, [key]: value}));
  const persist = async (reset = false) => {
    if (!data || disabled) return;
    setError(""); setSaving(true);
    try {
      if (reset) await api.resetSam3Settings(projectId);
      else await api.saveSam3Settings(projectId, Object.fromEntries(changed.map(k => [k, values[k]])));
      await onUpdated();
      setOpen(false);
    } catch (e) { setError(String((e as Error).message ?? e)); }
    finally { setSaving(false); }
  };

  return <>
    <Button variant="secondary" disabled={disabled} icon={<Settings2 className="size-4" />} onClick={() => setOpen(true)}>
      تنظیمات دقت SAM3
    </Button>
    {open && <div className="fixed inset-0 z-[130] overflow-y-auto bg-black/85 p-3 sm:p-8" dir="rtl">
      <div className="mx-auto max-w-4xl rounded-2xl border border-white/15 bg-[#15151f] p-4 shadow-2xl sm:p-6">
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-lg font-bold text-white"><SlidersHorizontal className="size-5 text-violet-300" /> تنظیمات دقیق SAM3</h2>
            <p className="mt-2 text-sm text-zinc-400">این تنظیمات مخصوص همین پروژه است. بعضی گزینه‌ها مربوط به مدل SAM3 و بعضی مخصوص پردازش ماسک در استودیو هستند.</p>
          </div>
          <button aria-label="بستن" onClick={() => setOpen(false)} className="rounded-lg bg-white/10 p-2 text-white"><X className="size-5" /></button>
        </div>
        <p className="mb-4 flex items-start gap-2 rounded-xl border border-amber-500/25 bg-amber-500/10 p-3 text-xs leading-6 text-amber-200">
          <AlertTriangle className="mt-1 size-4 shrink-0" /> بعد از تغییر تنظیمات مربوط به ترکینگ، ماسک‌های قبلی نامعتبر می‌شوند. لازم است شخص یا آبجکت‌های لباس را دوباره ردیابی کنی. افزایش رزولوشن می‌تواند RAM و زمان پردازش را به‌شدت بالا ببرد؛ 4K ورودی به معنی استنتاج مستقیم 4K داخل SAM3 نیست.
        </p>
        {loading && <p className="p-4 text-zinc-400">در حال دریافت پارامترهای SAM3…</p>}
        {data && <>
          <div className="mb-4 flex flex-wrap gap-2">
            {Object.entries(presets).map(([name, preset]) => <button key={name} onClick={() => setValues(v => ({ ...v, ...preset }))}
              className="rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-xs text-violet-100 hover:bg-violet-500/25">{name}</button>)}
            <button onClick={() => setValues(data.values)} className="rounded-lg border border-white/10 px-3 py-2 text-xs text-zinc-300">بازگردانی تغییرات ذخیره‌نشده</button>
            <button onClick={() => setAdvanced(x => !x)} className="ms-auto rounded-lg border border-white/10 px-3 py-2 text-xs text-zinc-300">{advanced ? "پنهان‌کردن گزینه‌های تخصصی" : "نمایش تمام پارامترهای تخصصی"}</button>
          </div>
          {Object.entries(data.groups).map(([group, title]) => {
            const fields = data.fields.filter(f => f.group === group && (advanced || group === "processing" || ["score_threshold", "new_det_threshold", "tracker_mask_threshold", "occlusion_min_iou", "occlusion_min_area", "image_mask_threshold"].includes(f.key)));
            if (!fields.length) return null;
            return <section key={group} className="mb-4 rounded-xl border border-white/10 bg-white/[0.025] p-3">
              <h3 className="mb-3 font-semibold text-violet-200">{title}</h3>
              <div className="grid gap-3 sm:grid-cols-2">
                {fields.map(f => <label key={f.key} className="min-w-0 rounded-lg bg-black/20 p-2">
                  <span className="block text-xs leading-5 text-zinc-200">{f.help}</span>
                  <code dir="ltr" className="my-1 block text-[11px] text-zinc-500">{f.key}</code>
                  {f.type === "boolean" ?
                    <input type="checkbox" checked={Boolean(values[f.key])} onChange={e => update(f.key, e.target.checked)} className="size-4 accent-violet-500" />
                    : f.type === "select" ?
                      <select value={String(values[f.key])} className={inputClass} onChange={e => update(f.key, e.target.value)}>
                        {(f.options ?? []).map(o => <option key={o} value={o}>{o}</option>)}
                      </select>
                      : <div className="flex items-center gap-2">
                        <input className={inputClass} dir="ltr" type="number" value={values[f.key] as number ?? f.default as number}
                          min={f.minimum ?? undefined} max={f.maximum ?? undefined}
                          step={f.type === "integer" ? 1 : 0.001}
                          onChange={e => update(f.key, Number(e.target.value))} />
                        <span className="shrink-0 text-[10px] text-zinc-500">{f.minimum}…{f.maximum}</span>
                      </div>}
                </label>)}
              </div>
            </section>;
          })}
          {error && <div role="alert" className="mb-3 rounded-lg bg-rose-500/10 p-3 text-sm text-rose-300">{error}</div>}
          <div className="sticky bottom-0 flex flex-wrap items-center gap-3 border-t border-white/10 bg-[#15151f] py-3">
            <Button variant="primary" disabled={!changed.length || saving || disabled} loading={saving} icon={<Save className="size-4" />} onClick={() => persist()}>
              ذخیره {changed.length ? `${changed.length} تغییر` : "تنظیمات"}
            </Button>
            <Button variant="secondary" disabled={saving || disabled} icon={<RotateCcw className="size-4" />} onClick={() => persist(true)}>
              بازنشانی به تنظیمات اولیه پروژه
            </Button>
            <span className="text-xs text-zinc-500">تنظیمات ۱ سکانس روی سکانس‌های دیگر اعمال نمی‌شود.</span>
          </div>
        </>}
      </div>
    </div>}
  </>;
}
