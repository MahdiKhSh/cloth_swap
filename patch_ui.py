from pathlib import Path
r=Path('/mnt/data/cloth_swap_review')
p=r/'studio-ui/src/api.ts';s=p.read_text();s=s.replace('export interface Candidate {','export interface MaskLayer {\n  name: string;\n  prompt: string;\n  operation: "add" | "subtract";\n}\n\nexport interface Candidate {');s=s.replace('  garment: { images: string[]; description: string };','  mask_layers: { rules: MaskLayer[]; ready: boolean };\n  garment: { images: string[]; description: string };');s=s.replace('  addGarment: (id: string, file: File) => {','  setMaskLayers: (id: string, layers: MaskLayer[]) => request<Project>("PUT", `${P(id)}/mask-layers`, { layers }),\n  buildMaskLayers: (id: string) => request<Job>("POST", `${P(id)}/mask-layers/build`),\n  addGarment: (id: string, file: File) => {');p.write_text(s)
p=r/'studio-ui/src/steps/StepActor.tsx';s=p.read_text();s=s.replace('type Polygons, type Prompt','type Polygons, type Prompt, type MaskLayer');s=s.replace('  const [part, setPart] = useState("actor");','''  const [part, setPart] = useState("actor");
  const [layerRules, setLayerRules] = useState<MaskLayer[]>(p.mask_layers?.rules ?? []);
  const [layerDirty, setLayerDirty] = useState(false);
  const [maskJobId, setMaskJobId] = useState<string | null>(() => {
    const j = lastJob("mask_layers");
    return j && (j.status === "running" || j.status === "queued") ? j.id : null;
  });
  const maskAction = useAction();
  const maskJob = useJob(maskJobId, async () => { masks.current.clear(); setLayerDirty(false); await refresh(); }, () => setMaskJobId(null));
  const setRules = (rules: MaskLayer[]) => { setLayerRules(rules); setLayerDirty(true); };
  const presetRules: MaskLayer[] = [
    { name: "clothes", prompt: "shirt, jacket, top or outerwear worn by the selected person", operation: "add" },
    { name: "pants", prompt: "pants or trousers worn by the selected person", operation: "add" },
    { name: "face", prompt: "face of the selected person", operation: "subtract" },
    { name: "hair", prompt: "hair of the selected person", operation: "subtract" },
    { name: "shoes", prompt: "shoes worn by the selected person", operation: "subtract" },
  ];
  const buildLayers = async () => {
    try {
      await maskAction.run(() => api.setMaskLayers(p.id, layerRules));
      if (layerRules.length === 0) { setLayerDirty(false); await refresh(); return; }
      const j = await maskAction.run(() => api.buildMaskLayers(p.id));
      if (j) setMaskJobId(j.id);
    } catch { /* useAction reports errors */ }
  };''');s=s.replace('const maskKey = `${p.updated}:${part}:${frame}`;','const maskKey = `${p.updated}:${part}:${frame}`;');s=s.replace('options={[{value: "actor", label: t("maskActor")}, {value: "face", label: t("maskFace")}, {value: "hair", label: t("maskHair")}]}', 'options={[{value: "actor", label: t("maskActor")}, {value: "face", label: t("maskFace")}, {value: "hair", label: t("maskHair")}, {value: "edit", label: "Edit +"}, {value: "protect", label: "Protect −"}]}');s=s.replace('          {dirty && <div', '''          <Card className="p-4">
            <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
              <span className="text-sm font-semibold text-white">{t("maskLayersTitle")}</span>
              <Button variant="secondary" onClick={() => setRules(presetRules)}>{t("maskPreset")}</Button>
            </div>
            <p className="mb-3 text-xs leading-5 text-zinc-400">{t("maskLayersHint")}</p>
            <div className="space-y-2">
              {layerRules.map((rule, i) => (
                <div key={i} className="flex gap-1.5">
                  <select value={rule.operation} aria-label="Mask operation" onChange={e => setRules(layerRules.map((r, j) => i === j ? {...r, operation: e.target.value as MaskLayer["operation"]} : r))}
                    className={clsx(inputClass, "w-14 px-1 text-center")}>
                    <option value="add">+</option><option value="subtract">−</option>
                  </select>
                  <div className="min-w-0 flex-1 space-y-1">
                    <input dir="ltr" aria-label="Layer name" className={clsx(inputClass,"w-full")} value={rule.name} placeholder="layer_name"
                      onChange={e => setRules(layerRules.map((r,j) => i===j ? {...r,name:e.target.value} : r))} />
                    <input dir="ltr" aria-label="SAM3 prompt" className={clsx(inputClass,"w-full")} value={rule.prompt} placeholder="English segmentation prompt"
                      onChange={e => setRules(layerRules.map((r,j) => i===j ? {...r,prompt:e.target.value} : r))} />
                  </div>
                  <button aria-label="Remove layer" onClick={() => setRules(layerRules.filter((_,j)=>i!==j))} className="self-start p-2 text-rose-300"><Trash2 className="size-4"/></button>
                </div>
              ))}
              <Button variant="secondary" icon={<PlusCircle className="size-4"/>} onClick={() => setRules([...layerRules,{name:`part_${layerRules.length+1}`,prompt:"",operation:"add"}])}>{t("maskAddLayer")}</Button>
            </div>
            <p className="mt-3 text-xs" dir="ltr">{layerRules.filter(x=>x.operation==="add").map(x=>x.name).join(" + ") || "∅"} − ({layerRules.filter(x=>x.operation==="subtract").map(x=>x.name).join(" + ") || "∅"})</p>
            {layerDirty && <p className="mt-2 text-xs text-amber-300">{t("maskUnsaved")}</p>}
            {p.mask_layers?.ready && !layerDirty && <p className="mt-2 text-xs text-emerald-300">{t("maskReady")}</p>}
            {maskJob && <JobPanel job={maskJob}/>}
            <Button variant="primary" loading={!!maskJobId || maskAction.busy} disabled={!p.actor.tracked || layerRules.some(r=>!r.name.trim() || !r.prompt.trim())}
              onClick={buildLayers}>{t("maskBuild")}</Button>
          </Card>
          {dirty && <div''');p.write_text(s)
p=r/'studio-ui/src/i18n.ts';s=p.read_text();# add keys before detect entries in both locales
s=s.replace('  detect: "تشخیص خودکار افراد",','''  maskLayersTitle: "ماسک ترکیبی (+ / −)",
  maskPreset: "نمونه لباس و شلوار − صورت و مو و کفش",
  maskLayersHint: "برای هر ناحیه یک پرامپت انگلیسی SAM3 بنویسید. + برای تغییر و − برای محافظت است. پس از ردیابی شخص، ماسک نهایی را بسازید.",
  maskAddLayer: "افزودن لایه",
  maskBuild: "ساخت و ردیابی ماسک‌ها",
  maskReady: "ماسک نهایی آماده است؛ Edit + و Protect − را مشاهده کنید.",
  maskUnsaved: "تغییرات لایه‌ها هنوز اعمال نشده‌اند.",
  detect: "تشخیص خودکار افراد",''');s=s.replace('  detect: "Detect people",','''  maskLayersTitle: "Combined edit mask (+ / −)",
  maskPreset: "Clothes + pants − face − hair − shoes",
  maskLayersHint: "Use an English SAM3 prompt for each part. + edits; − protects. Build masks after tracking the person.",
  maskAddLayer: "Add layer",
  maskBuild: "Build and track masks",
  maskReady: "Mask ready; preview Edit + or Protect −.",
  maskUnsaved: "Layer changes have not been applied yet.",
  detect: "Detect people",''');p.write_text(s)
