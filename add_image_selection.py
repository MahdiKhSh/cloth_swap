from pathlib import Path
r=Path(__file__).parent
p=r/'cloth_swap/studio/service.py';s=p.read_text();s=s.replace('            if not name or not prompt or op not in ("add", "subtract"):\n                raise ValueError("Each mask layer needs name, English prompt, and add/subtract")','''            selection = rule.get("selection")
            if not name or op not in ("add", "subtract") or (not prompt and not selection):
                raise ValueError("Each mask layer needs a name, add/subtract, and a text prompt or image selection")
            if selection:
                t = int(selection.get("frame", -1))
                start, end = p.state["trim"]["start"], p.state["trim"]["end"]
                if not start <= t < end:
                    raise ValueError("Mask selection frame must be inside video trim")
                points = selection.get("points") or []
                labels = selection.get("labels") or []
                box = selection.get("box")
                if not points and not box:
                    raise ValueError("Image selection must have at least a point or a box")
                if len(points) != len(labels) or any(len(pt) != 2 for pt in points) or (box is not None and len(box) != 4):
                    raise ValueError("Invalid image selection points or box")
                selection = {"frame": t, "points": [[float(v) for v in pt] for pt in points],
                             "labels": [int(v) for v in labels],
                             "box": [float(v) for v in box] if box else None}
                if any(v not in (0, 1) for v in selection["labels"]):
                    raise ValueError("Image selection labels must be 0 or 1")''');s=s.replace('clean.append({"name": name, "prompt": prompt, "operation": op})','clean.append({"name": name, "prompt": prompt, "operation": op, "selection": selection})');p.write_text(s)
p=r/'cloth_swap/studio/worker.py';s=p.read_text();s=s.replace('    from ..segmentation import track_concepts\n','    from ..segmentation import track_concepts, track_prompts\n');s=s.replace('''        tracks = track_concepts(frames, [rule["prompt"]], cfg)
        mask = masks_within(tracks, rule["prompt"], near) & actor''','''        if rule.get("selection"):
            prompt = dict(rule["selection"])
            prompt["frame"] -= a["start"]
            # One image selection is propagated over the source frames; WAN runs only once.
            mask = track_prompts(frames, [prompt], cfg) & near & actor
        else:
            tracks = track_concepts(frames, [rule["prompt"]], cfg)
            mask = masks_within(tracks, rule["prompt"], near) & actor''');p.write_text(s)
p=r/'cloth_swap/studio/api.py';s=p.read_text().replace('class MaskLayer(BaseModel):\n    name: str\n    prompt: str\n    operation: str','class MaskLayer(BaseModel):\n    name: str\n    prompt: str = ""\n    operation: str\n    selection: Prompt | None = None');p.write_text(s)
p=r/'studio-ui/src/api.ts';s=p.read_text().replace('  operation: "add" | "subtract";\n}', '  operation: "add" | "subtract";\n  selection?: Prompt | null;\n}');p.write_text(s)
p=r/'studio-ui/src/steps/StepActor.tsx';s=p.read_text();s=s.replace('  const [layerDirty, setLayerDirty] = useState(false);','''  const [layerDirty, setLayerDirty] = useState(false);
  const [activeLayer, setActiveLayer] = useState<number | null>(null);
  const [layerPreview, setLayerPreview] = useState<{ key: string; polygons: Polygons } | null>(null);''');s=s.replace('  const current = prompts[frame];','''  const current = prompts[frame];
  const activeSelection = activeLayer === null ? null : layerRules[activeLayer]?.selection;
  const selectionOnFrame = activeSelection?.frame === frame ? activeSelection : null;
  const selectionKey = JSON.stringify(selectionOnFrame ?? null);
  const debouncedSelection = useDebounced(selectionKey, 180);
  useEffect(() => {
    if (!selectionOnFrame || debouncedSelection !== selectionKey || (!selectionOnFrame.points.length && !selectionOnFrame.box)) return;
    let stopped = false;
    api.segment(p.id, selectionOnFrame).then(result => {
      if (!stopped) setLayerPreview({key: selectionKey, polygons: result.polygons});
    }).catch(() => {});
    return () => { stopped = true; };
  }, [debouncedSelection, selectionKey, p.id]); // eslint-disable-line react-hooks/exhaustive-deps
  const editLayerSelection = (fn: (pr: Prompt) => Prompt) => {
    if (activeLayer === null) return;
    const currentSelection = layerRules[activeLayer]?.selection;
    const previous = currentSelection?.frame === frame ? currentSelection : {frame, points: [], labels: [], box: null};
    const next = fn(previous);
    setRules(layerRules.map((rule, i) => i === activeLayer ? {...rule, selection: next} : rule));
    setLayerPreview(null);
  };''');s=s.replace('    if (part !== "actor") return;','    if (part !== "actor" || activeLayer !== null) return;');s=s.replace('  const showPeople = part === "actor" && people', '  const showPeople = activeLayer === null && part === "actor" && people');s=s.replace('  if (showPeople) {','''  if (activeLayer !== null && layerPreview?.key === selectionKey && selectionOnFrame) {
    layers.push({polygons: layerPreview.polygons, className: "mask-fill"});
  } else if (showPeople) {''');s=s.replace('  } else if (part === "actor" && !isEmpty(current)', '  } else if (activeLayer === null && part === "actor" && !isEmpty(current)');s=s.replace('  } else if ((part !== "actor" || isEmpty(current)) && p.actor.tracked', '  } else if (activeLayer === null && (part !== "actor" || isEmpty(current)) && p.actor.tracked');s=s.replace('tool={part !== "actor" || showPeople ? null : tool}', 'tool={activeLayer !== null ? tool : (part !== "actor" || showPeople ? null : tool)}');s=s.replace('points={(part === "actor" ? current?.points ?? [] : []).map(([x, y], i) => ({ x, y, label: current!.labels[i] ?? 1 }))}', 'points={(activeLayer !== null ? selectionOnFrame?.points ?? [] : part === "actor" ? current?.points ?? [] : []).map(([x, y], i) => ({ x, y, label: (activeLayer !== null ? selectionOnFrame?.labels[i] : current?.labels[i]) ?? 1 }))}');s=s.replace('box={part === "actor" ? current?.box ?? null : null}', 'box={activeLayer !== null ? selectionOnFrame?.box ?? null : part === "actor" ? current?.box ?? null : null}');s=s.replace('onPoint={(x, y, label) => edit((pr)', 'onPoint={(x, y, label) => (activeLayer !== null ? editLayerSelection : edit)((pr)');s=s.replace('onBox={(box) => edit((pr)', 'onBox={(box) => (activeLayer !== null ? editLayerSelection : edit)((pr)');s=s.replace('              {layerRules.map((rule, i) => (\n                <div key={i} className="flex gap-1.5">','''              {layerRules.map((rule, i) => (
                <div key={i} className="flex flex-col gap-1.5 rounded-lg border border-white/10 p-2">
                  <button type="button" className={clsx("rounded-lg px-2 py-2 text-start text-xs", activeLayer === i ? "bg-violet-500/25 text-white" : "bg-white/5 text-zinc-300")}
                    onClick={() => { setActiveLayer(activeLayer === i ? null : i); if (rule.selection) player.setFrame(rule.selection.frame); player.setPlaying(false); }}>
                    {activeLayer === i ? "✓ " : "◎ "}Select on image: {rule.name} {rule.selection ? `(frame ${rule.selection.frame})` : ""}
                  </button>
                  <div className="flex gap-1.5">''');s=s.replace('                </div>\n              ))}\n              <Button variant="secondary" icon={<PlusCircle', '''                  </div>
                  {rule.selection && <button type="button" className="text-start text-xs text-rose-300" onClick={() => setRules(layerRules.map((r,j) => i === j ? {...r, selection: null} : r))}>Clear image selection</button>}
                </div>
              ))}
              <Button variant="secondary" icon={<PlusCircle''');s=s.replace('disabled={!p.actor.tracked || layerRules.some(r=>!r.name.trim() || !r.prompt.trim())}', 'disabled={!p.actor.tracked || layerRules.some(r=>!r.name.trim() || (!r.prompt.trim() && !r.selection))}');s=s.replace('            <p className="mb-3 text-xs leading-5 text-zinc-400">{t("maskLayersHint")}</p>', '''            <p className="mb-3 text-xs leading-5 text-zinc-400">Choose a layer, then click/box an object on the image (right click removes). Preview is computed on that single frame. Each selection is tracked across the source video only when you build masks. No separate videos are generated. Text prompts are optional for layers with image selection.</p>
            {activeLayer !== null && <button type="button" className="mb-2 rounded-lg bg-violet-500/25 px-3 py-2 text-xs text-white" onClick={() => setActiveLayer(null)}>Done selecting: {layerRules[activeLayer]?.name}</button>}''');p.write_text(s)
