"""Asks questions to the on-device app over USB and times them, by driving
its WebView through the Chrome DevTools protocol (debug builds enable
WebView debugging -- see MainActivity). Calls the same bridgeAsk() the
"Get guidance" button calls, so the timing is the phone's real answer time
(excluding only the page re-render).

Usage: python scripts/phone_ask.py "question 1" "question 2" ...
       python scripts/phone_ask.py --eval   (all evaluation questions, all languages)
Requires the app to be open on a USB-connected phone with USB debugging on.
"""

import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
ADB = str(ROOT / "android-sdk" / "platform-tools" / "adb.exe")
sys.stdout.reconfigure(encoding="utf-8")

JS = """(async () => {
  const t = performance.now();
  const d = await bridgeAsk(%s, "auto");
  return JSON.stringify({ms: Math.round(performance.now() - t), cached: !!d.cached, lang: d.answer_language,
                         abstained: d.abstained, error: d.error || null, text: (d.text || "").slice(0, 140)});
})()"""


def connect_url():
    pid = subprocess.run([ADB, "shell", "pidof", "com.disasterassistant.app"], capture_output=True, text=True).stdout.strip()
    if not pid:
        sys.exit("app is not running on the phone")
    subprocess.run([ADB, "forward", "tcp:9222", f"localabstract:webview_devtools_remote_{pid}"], check=True,
                   capture_output=True)
    for _ in range(60):
        try:
            pages = json.load(urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=3))
            for p in pages:
                if p.get("url", "").startswith("file:///android_asset/www/index.html"):
                    return p["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(2)
    sys.exit("app page not found (still loading the model?)")


async def ask_all(questions):
    async with websockets.connect(connect_url(), max_size=2 ** 24) as ws:
        n = 0

        async def call(expr):
            nonlocal n
            n += 1
            await ws.send(json.dumps({"id": n, "method": "Runtime.evaluate",
                                      "params": {"expression": expr, "awaitPromise": True, "returnByValue": True}}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == n:
                    return msg["result"]["result"].get("value")

        while await call("typeof AndroidBridge") != "object":
            await asyncio.sleep(2)
        results = []
        for q in questions:
            r = json.loads(await call(JS % json.dumps(q)))
            r["q"] = q
            results.append(r)
            tag = "INSTANT" if r["cached"] else ("refused" if r["abstained"] else "live AI")
            print(f"{r['ms'] / 1000:6.2f}s  {tag:8} [{r['lang']}] {q[:55]}")
            if r["error"]:
                print("        error:", r["error"])
        return results


def main():
    if sys.argv[1:] == ["--eval"]:
        qs = []
        for name in ["questions_v1.json", "questions_v2_heldout.json"]:
            for it in json.loads((ROOT / "data/evaluation" / name).read_text(encoding="utf-8"))["items"]:
                qs += list(it["q"].values())
    else:
        qs = sys.argv[1:]
    asyncio.run(ask_all(qs))


if __name__ == "__main__":
    main()
