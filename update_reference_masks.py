from pathlib import Path
base=Path(__file__).parent

def change(rel,old,new):
 p=base/rel;s=p.read_text();assert old in s, (rel,old[:100]);p.write_text(s.replace(old,new))
change('cloth_swap/studio/api.py','class SegmentRequest(Prompt):\n    pass','class SegmentRequest(Prompt):\n    candidate: str | None = None')
change('cloth_swap/studio/api.py','    selection: Prompt | None = None  # add | subtract','    selection: dict | None = None  # selected reference candidate + point/box hints')
change('cloth_swap/studio/api.py','body.frame, body.points, body.labels, body.box)','body.frame, body.points, body.labels, body.box, body.candidate)')
change('cloth_swap/studio/service.py','def segment_preview(self, p: Project, t: int, points, labels, box) -> dict:\n        image = self.frame(p, t)','''def segment_preview(self, p: Project, t: int, points, labels, box, candidate=None) -> dict:
        if candidate is not None:
            if p.state["keyframe"].get("chosen") != candidate:
                raise ValueError("Choose the reference image before segmenting")
            selected = next((c for c in p.state["keyframe"]["candidates"] if c["id"] == candidate), None)
            if selected is None or selected["frame"] != t:
                raise ValueError("Reference selection must use the selected keyframe")
            image = np.asarray(Image.open(p.path("02_keyframe", "candidates", candidate + ".jpg")).convert("RGB"))
        else:
            image = self.frame(p, t)''')
change('cloth_swap/studio/service.py','''                selection = {"frame": t, "points": [[float(v) for v in pt] for pt in points],
                             "labels": [int(v) for v in labels],
                             "box": [float(v) for v in box] if box else None}''','''                candidate = selection.get("candidate")
                if candidate:
                    chosen = p.state["keyframe"].get("chosen")
                    selected = next((c for c in p.state["keyframe"]["candidates"] if c["id"] == candidate), None)
                    if chosen != candidate or not selected or selected["frame"] != t:
                        raise ValueError("Mask selection is not on the currently chosen reference image")
                selection = {"frame": t, "points": [[float(v) for v in pt] for pt in points],
                             "labels": [int(v) for v in labels],
                             "box": [float(v) for v in box] if box else None,
                             "candidate": candidate}''')
change('cloth_swap/studio/service.py','''        p.update("keyframe", {"chosen": cid, "frame": cand["frame"]})''','''        if p.state["keyframe"].get("chosen") != cid and p.state["mask_layers"].get("rules"):
            p.update("mask_layers", {"ready": False})
        p.update("keyframe", {"chosen": cid, "frame": cand["frame"]})''')
change('studio-ui/src/api.ts','''  selection?: Prompt | null;''','''  selection?: (Prompt & { candidate?: string | null }) | null;''')
change('studio-ui/src/api.ts','''  segment: (id: string, prompt: Prompt) => request<{ polygons: Polygons; area: number }>("POST", `${P(id)}/segment`, prompt),''','''  segment: (id: string, prompt: Prompt & { candidate?: string | null }) => request<{ polygons: Polygons; area: number }>("POST", `${P(id)}/segment`, prompt),''')
# Remove mask editing and misleading face/hair toggles from actor step. Retain the person tracker controls.
p=base/'studio-ui/src/steps/StepActor.tsx';s=p.read_text();a=s.index('          <Card className="p-4">\n            <div className="mb-2 flex flex-wrap');b=s.index('          {dirty &&',a);s=s[:a]+s[b:];s=s.replace('''          <Segmented value={part} onChange={setPart} options={[{value: "actor", label: t("maskActor")}, {value: "face", label: t("maskFace")}, {value: "hair", label: t("maskHair")}, {value: "edit", label: "Edit +"}, {value: "protect", label: "Protect −"}]} />
          {part !== "actor" && <p className="text-xs text-zinc-400">{t("maskPartsHint")}</p>}''','''          <p className="text-xs text-zinc-400">این مرحله فقط برای انتخاب و ردیابی شخص در ویدئو است. ماسک‌های قابل‌تغییر و محافظت‌شده را بعداً روی عکس رفرنس تعیین کنید.</p>''');p.write_text(s)
