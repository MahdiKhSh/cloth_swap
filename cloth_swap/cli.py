"""Command line entry point: python -m cloth_swap --video in.mp4 --garment shirt.png --out out.mp4"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import load_config, set_dotted
from .pipeline import STAGES, Pipeline


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="cloth_swap", description="Replace an actor's clothes in a video (SAM3 + Qwen-Image-Edit + SCAIL-2).")
    p.add_argument("--video", required=True, help="input video")
    p.add_argument("--out", required=True, help="output video")
    p.add_argument("--garment", help="photo of the new garment (flat lay or worn)")
    p.add_argument("--garment-desc", help="text description of the new garment (also used in the SCAIL-2 prompt)")
    p.add_argument("--subject-desc", help="who the actor is, for the SCAIL-2 prompt, e.g. 'A tall man with short black hair'")
    p.add_argument("--scene-desc", help="what happens in the shot, for the SCAIL-2 prompt")
    p.add_argument("--prompt", help="full SCAIL-2 prompt (overrides the three descriptions)")
    p.add_argument("--keyframe-image", help="an already edited frame to use instead of running Qwen (frame index via --keyframe-frame)")
    p.add_argument("--keyframe-frame", type=int, help="index (within the processed clip) of the keyframe")
    p.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"), help="click on the actor (original pixels)")
    p.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="box around the actor")
    p.add_argument("--select-frame", type=int, help="frame where --point/--box are given")
    p.add_argument("--start-frame", type=int)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--config", help="YAML file merged over configs/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value, e.g. scail.steps=40")
    p.add_argument("--work-dir", help="intermediate files (default: work/<video name>)")
    p.add_argument("--from-stage", choices=STAGES, help="recompute from this stage on")
    p.add_argument("--stop-after", choices=STAGES, help="stop after this stage (e.g. keyframe, to inspect it)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config, args.set)
    cli_values = {
        "prompt.garment": args.garment_desc,
        "prompt.subject": args.subject_desc,
        "prompt.scene": args.scene_desc,
        "scail.prompt": args.prompt,
        "keyframe.frame": args.keyframe_frame,
        "actor.point": list(args.point) if args.point else None,
        "actor.box": list(args.box) if args.box else None,
        "actor.select_frame": args.select_frame,
        "video.start_frame": args.start_frame,
        "video.max_frames": args.max_frames,
    }
    for key, value in cli_values.items():
        if value is not None:
            set_dotted(cfg, key, value)
    if not (args.garment or args.garment_desc or args.keyframe_image):
        raise SystemExit("Give --garment, --garment-desc or --keyframe-image")

    work_dir = args.work_dir or str(Path("work") / Path(args.video).stem)
    Pipeline(cfg, args.video, args.out, work_dir, garment=args.garment,
             keyframe_image=args.keyframe_image).run(args.from_stage, args.stop_after)
    return 0
