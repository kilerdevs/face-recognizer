#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# main.py  –  entry point: bootstrap → GUI
# ---------------------------------------------------------------------------

import logging
import sys
import warnings

# Suppress noisy warnings from the experimental NumPy MINGW-W64 build
# (Python 3.14 ships no stable numpy wheel on Windows; these are harmless here).
warnings.filterwarnings("ignore", message=".*MINGW-W64.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# Root logger at DEBUG so the GUI handler sees everything;
# individual loggers can raise their own levels.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def _main():
    # ---- auto-install missing packages + collect env info ----
    try:
        import bootstrap
        env_info = bootstrap.run(auto_install=True)
    except Exception as exc:
        print(f"Bootstrap warning: {exc}")
        env_info = {}

    # ---- launch GUI ----
    try:
        from gui import run
        run(env_info=env_info)
    except ImportError as exc:
        print(
            f"\nFATAL: cannot import GUI module: {exc}\n"
            f"Make sure PyQt5 is installed:\n"
            f"  pip install PyQt5>=5.15.0\n",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        import traceback
        print(f"\nFATAL: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    _main()
