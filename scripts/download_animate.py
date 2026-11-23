"""Resume the compact, verified Animate download without fetching another T5 or VAE."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).with_name('download_animate_compact.py')), run_name='__main__')
