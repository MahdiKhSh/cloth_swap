from pathlib import Path
R=Path('/mnt/data/cloth_swap_review')
def change(path, old, new):
 p=R/path;s=p.read_text();assert old in s,(path,old[:90]);p.write_text(s.replace(old,new))
change('cloth_swap/studio/store.py','    "garment": {"images": [], "description": ""},','    "mask_layers": {"rules": [], "ready": False},\n    "garment": {"images": [], "description": ""},')
change('cloth_swap/studio/api.py','class KeyframeRequest(BaseModel):', '''class MaskLayer(BaseModel):
    name: str
    prompt: str
    operation: str  # add | subtract


class MaskLayersRequest(BaseModel):
    layers: list[MaskLayer]


class KeyframeRequest(BaseModel):''')
change('cloth_swap/studio/api.py','    # ---------------------------------------------------------------- garment','''    @app.put("/api/projects/{pid}/mask-layers")
    def mask_layers(pid: str, body: MaskLayersRequest):
        return service.set_mask_layers(project(pid), [v.model_dump() for v in body.layers]).to_dict()

    @app.post("/api/projects/{pid}/mask-layers/build")
    def build_mask_layers(pid: str):
        return service.build_mask_layers(project(pid)).to_dict()

    # ---------------------------------------------------------------- garment''')
change('cloth_swap/studio/api.py','if part not in {"actor", "face", "hair"}:','if part not in {"actor", "face", "hair", "edit", "protect"}:')
change('cloth_swap/studio/service.py','from .. import masks as mk','from .. import masks as mk\nfrom ..mask_layers import compose')
change('cloth_swap/studio/service.py','if part not in {"actor", "face", "hair"}:','if part not in {"actor", "face", "hair", "edit", "protect"}:')
change('cloth_swap/studio/service.py','            p.update("actor", {"tracked": True, "visible": visible, "trim": [start, end]})','            p.update("actor", {"tracked": True, "visible": visible, "trim": [start, end]})\n            p.update("mask_layers", {"ready": False})')
change('cloth_swap/studio/service.py','    # ------------------------------------------------------------------ garment\n', '''    # ------------------------------------------------------------------ layered edit masks
    def set_mask_layers(self, p: Project, rules: list[dict]) -> Project:
        clean = []
        for rule in rules:
            name = (rule.get("name") or "").strip().lower()
            prompt = (rule.get("prompt") or "").strip()
            op = rule.get("operation")
            if not name or not prompt or op not in ("add", "subtract"):
                raise ValueError("Each mask layer needs name, English prompt, and add/subtract")
            if re.search(r"[؀-ۿ]", prompt):
                raise ValueError("english_only")
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name):
                raise ValueError("Layer names must be lowercase letters, digits or underscores")
            if name in (r["name"] for r in clean):
                raise ValueError("Duplicate mask layer name")
            clean.append({"name": name, "prompt": prompt, "operation": op})
        if len(clean) > 12:
            raise ValueError("Maximum 12 layers")
        p.update("mask_layers", {"rules": clean, "ready": False})
        return p

    def build_mask_layers(self, p: Project) -> Job:
        if not p.state["actor"]["tracked"]:
            raise ValueError("Track the person before segmenting parts")
        rules = p.state["mask_layers"]["rules"]
        if not rules or not any(r["operation"] == "add" for r in rules):
            raise ValueError("Add at least one positive mask layer")
        start, end = p.state["trim"]["start"], p.state["trim"]["end"]

        def run(job: Job):
            actor = self.masks(p, "actor")
            if actor is None or actor.shape[0] != end - start:
                raise ValueError("Actor tracking must match the current trim")
            result = self.worker.call("mask_layers", {
                "source": str(p.source), "start": start, "count": end - start,
                "rules": rules, "actor_path": str(p.path("01_masks", "actor.npz")),
                "output_dir": str(p.path("01_masks", "layers", ".keep").parent),
                "edit_path": str(p.path("01_masks", "edit.npz")),
                "protect_path": str(p.path("01_masks", "protect.npz"))}, job)
            p.update("mask_layers", {"ready": True})
            self._mask_cache = None
            return result

        return self.jobs.submit(p.id, "mask_layers", "mask_layers", run)

    # ------------------------------------------------------------------ garment
''')
change('cloth_swap/studio/service.py','        if not p.state["keyframe"]["chosen"]:\n            raise ValueError("Choose a keyframe first")','        if not p.state["keyframe"]["chosen"]:\n            raise ValueError("Choose a keyframe first")\n        if p.state["mask_layers"]["rules"] and not p.state["mask_layers"]["ready"]:\n            raise ValueError("Build the layered edit mask before generation")')
change('cloth_swap/studio/worker.py','TASKS = {"track": _track, "variants": _variants, "finish_upload": _finish_upload}', '''def _mask_layers(cache: dict, cfg: dict, a: dict, progress) -> dict:
    from pathlib import Path
    from .. import masks as mk
    from ..mask_layers import compose
    from ..segmentation import track_concepts
    from ..tracks import masks_within
    from ..video_io import read_frames

    frames = read_frames(a["source"], a["start"], a["count"])
    actor = mk.load_masks(a["actor_path"])
    if actor.shape != frames.shape[:3]:
        raise ValueError("Actor mask and video shape mismatch")
    rules = a["rules"]
    layers = {}
    # Run each prompt in its own session so object IDs and prompt association never mix.
    # Restrict every result to the selected person, not similarly dressed passers-by.
    near = mk.temporal_union(actor, 1)
    output = Path(a["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for i, rule in enumerate(rules):
        progress(0.03 + 0.80 * i / len(rules), f"segmenting:{rule['name']}")
        tracks = track_concepts(frames, [rule["prompt"]], cfg)
        mask = masks_within(tracks, rule["prompt"], near) & actor
        layers[rule["name"]] = mask
        mk.save_masks(output / f"{rule['name']}.npz", mask)
    edit, protect = compose(actor, layers, rules, protection_px=0)
    mk.save_masks(a["edit_path"], edit)
    mk.save_masks(a["protect_path"], protect)
    progress(0.95, "mask_ready")
    return {"edited_pixels": int(edit.sum()), "protected_pixels": int(protect.sum()),
            "coverage": round(float(edit.mean()), 5),
            "layers": {name: int(mask.sum()) for name, mask in layers.items()}}


TASKS = {"track": _track, "variants": _variants, "finish_upload": _finish_upload,
         "mask_layers": _mask_layers}''')
change('cloth_swap/pipeline.py','        actor = self.load_mask("actor")\n\n        new_garment = None','        actor = self.load_mask("actor")\n        edit = self.load_mask("edit") if self.cfg.get("mask_layers_enabled") else None\n        protected = self.load_mask("protect") if self.cfg.get("mask_layers_enabled") else None\n\n        new_garment = None')
change('cloth_swap/pipeline.py','                                content=m.get("content_box"))','                                content=m.get("content_box"), edit=edit, protected=protected)')
change('cloth_swap/studio/service.py','        comp = s["composite"]','        cfg["mask_layers_enabled"] = bool(s["mask_layers"]["rules"] and s["mask_layers"]["ready"])\n        comp = s["composite"]')
change('cloth_swap/composite.py','                 new_garment: np.ndarray | None = None, content=None):','                 new_garment: np.ndarray | None = None, content=None,\n                 edit: np.ndarray | None = None, protected: np.ndarray | None = None):')
change('cloth_swap/composite.py','        self.actor = actor','        self.actor = actor\n        self.edit = edit\n        self.protected = protected')
change('cloth_swap/composite.py','        return alpha * self.content if self.content is not None else alpha','        if self.edit is not None:\n            # For layered edit, union/feather must not bring back protected pixels.\n            alpha = alpha * mk.dilate(self.edit[t], self.dilate_px)\n        if self.protected is not None:\n            alpha[self.protected[t]] = 0.0\n        return alpha * self.content if self.content is not None else alpha')
change('cloth_swap/composite.py','        if self.mode == "full":','        if self.edit is not None:\n            region = mk.union_window(self.edit, t, self.radius)\n            region = mk.fill_holes(mk.dilate(region, self.dilate_px))\n            return mk.feather(region, self.feather_px)\n        if self.mode == "full":')
change('cloth_swap/animate_stage.py',"    mk.save_masks(folder / 'actor.npz', actor)","    mk.save_masks(folder / 'actor.npz', actor)\n    if cfg.get('mask_layers_enabled'):\n        root_masks = folder.parent / '01_masks'\n        for name in ('edit', 'protect'):\n            src = root_masks / f'{name}.npz'\n            if not src.exists():\n                raise ValueError(f'Missing layered mask: {src}')\n            import shutil\n            shutil.copyfile(src, folder / f'{name}.npz')")
change('cloth_swap/animate_runner.py','from . import masks as mk','from . import masks as mk\nfrom .mask_layers import expand_edit')
change('cloth_swap/animate_runner.py',"    masks = mk.load_masks(folder / 'actor.npz')","    masks = mk.load_masks(folder / 'actor.npz')\n    if (folder / 'edit.npz').exists():\n        edit = mk.load_masks(folder / 'edit.npz')\n        protect = mk.load_masks(folder / 'protect.npz')\n        if edit.shape != masks.shape or protect.shape != masks.shape:\n            raise ValueError('Layered mask dimensions do not match actor')\n        # Pose uses original actor, but WAN replacement is restricted to edit pixels.\n        edit = expand_edit(edit, protect, dilate_px=max(1, round(masks.shape[1]*0.015)))\n    else:\n        edit = None")
change('cloth_swap/animate_runner.py',"        mask = cv2.dilate(actor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))","        mask = (cv2.resize(edit[i].astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)\n                if edit is not None else\n                cv2.dilate(actor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))))\n        if edit is not None:\n            protect_small = cv2.resize(protect[i].astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)\n            mask[protect_small > 0] = 0")
change('cloth_swap/pipeline.py','        inputs = scail2_stage.prepare_inputs(\n','        inputs = scail2_stage.prepare_inputs(\n')
# SCAIL replace mask use composed region; call uses actor keyword positional unchanged.
change('cloth_swap/pipeline.py','self.load_mask("actor"),\n            self._keyframes()', '(self.load_mask("edit") if self.cfg.get("mask_layers_enabled") else self.load_mask("actor")),\n            self._keyframes()')
change('cloth_swap/scail2_stage.py','    region = mk.temporal_union(actor, scfg["replace_temporal"])','    region = mk.temporal_union(actor, scfg["replace_temporal"])\n    protected_path = out_dir.parent / "01_masks" / "protect.npz"\n    protected = mk.load_masks(protected_path) if cfg.get("mask_layers_enabled") else None')
change('cloth_swap/scail2_stage.py','            yield mk.scail_mask_frame(small, "white")','            if protected is not None:\n                small &= ~mk.resize_mask(protected[i], (width, height))\n            yield mk.scail_mask_frame(small, "white")')
