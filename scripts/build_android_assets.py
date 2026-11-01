"""Copies api/static/ into android-app/app/src/main/assets/www/, so the
Android app bundles the web UI locally instead of fetching it from the
server on every launch. Run this before rebuilding the APK whenever
api/static/ changes.

Keeps the exact same relative folder structure (index.html at the root,
a static/ subfolder beside it) on purpose: index.html's asset references
are plain relative paths ("static/app.js", not "/static/app.js"), which
resolve correctly both when the page is served by FastAPI from its own
root ("/" + "static/app.js" = "/static/app.js") and when loaded from
file:///android_asset/www/index.html in the bundled app (same relative
resolution, no server-side routing changes needed either way).
"""

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "disaster_assistant" / "api" / "static"
DEST = ROOT / "android-app" / "app" / "src" / "main" / "assets" / "www"


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    (DEST / "static").mkdir(parents=True)

    shutil.copy2(SRC / "index.html", DEST / "index.html")
    for name in ["app.js", "style.css", "manifest.json", "icon.svg", "sw.js"]:
        shutil.copy2(SRC / name, DEST / "static" / name)

    print(f"Copied {SRC} -> {DEST}")


if __name__ == "__main__":
    main()
