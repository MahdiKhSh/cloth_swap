"""Start the studio: python -m cloth_swap.studio [--config configs/local.yaml] [--port 7860]"""
import argparse
import logging
import threading
import webbrowser
from pathlib import Path

import uvicorn

from ..config import PROJECT_ROOT
from .api import create_app
from .service import StudioService


def main() -> None:
    p = argparse.ArgumentParser(prog="cloth_swap.studio")
    default_cfg = PROJECT_ROOT / "configs" / "local.yaml"
    p.add_argument("--config", default=str(default_cfg) if default_cfg.exists() else None)
    p.add_argument("--data", default=str(PROJECT_ROOT / "studio_data"), help="folder for projects")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    service = StudioService(Path(args.data), args.config)
    threading.Thread(target=service.warmup, daemon=True).start()
    app = create_app(service)
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Cloth Swap Studio: {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
