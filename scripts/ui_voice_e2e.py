"""Real-browser voice test. Chrome is launched with a FAKE microphone that plays
a synthesised spoken question, so the genuine record -> upload -> transcribe ->
answer -> audio path runs in a browser against the RUNNING server.

  A) an insecure origin (plain http on the LAN address) must explain, not fail
  B) the secure origin (https) must record, answer, and fetch audio

Usage: python scripts/ui_voice_e2e.py --lan-ip 10.243.106.166
(synthetic speech: proves the path works, says nothing about real-voice accuracy)
"""

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_e2e  # noqa: E402  (reuses the CDP helper, check(), screenshots)

PORT = 9444


async def with_browser(wav: str, fn):
    profile = tempfile.mkdtemp(prefix="ui_voice_")
    chrome = subprocess.Popen(
        [ui_e2e.CHROME, "--headless=new", "--disable-gpu", f"--remote-debugging-port={PORT}",
         "--ignore-certificate-errors", "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
         f"--use-file-for-fake-audio-capture={wav}", f"--user-data-dir={profile}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        ws_url = None
        for _ in range(40):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            c = ui_e2e.CDP(ws)
            pump = asyncio.create_task(c.pump())
            for d in ("Page", "Runtime", "Log"):
                await c.call(f"{d}.enable")
            await c.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=2, mobile=True)
            await fn(c)
            pump.cancel()
    finally:
        chrome.terminate()


async def main(lan_ip, secure_url):
    from disaster_assistant.audio.tts import synthesize_to_wav

    wav = str(Path(tempfile.mkdtemp(prefix="fakemic_")) / "q.wav")
    synthesize_to_wav("What should I do during an earthquake?", wav, "en")
    check = ui_e2e.check

    async def insecure(c):
        await c.call("Page.navigate", url=f"http://{lan_ip}:8001/?theme=dark")
        await asyncio.sleep(2.5)
        check("[A] plain-http LAN origin is NOT a secure context", await c.js("window.isSecureContext") is False)
        await c.js("document.querySelector('.tab-btn[data-tab=\"voice\"]').click(); true")
        note = await c.js("(()=>{const n=document.getElementById('voice-secure-note');return n.hidden?null:n.innerText})()")
        check("Speak tab warns about the secure address up front", note and "8443" in note, str(note))
        await c.js("document.getElementById('record-btn').click(); true")
        await asyncio.sleep(1)
        msg = await c.js("document.getElementById('voice-status').textContent")
        check("tapping the mic explains the fix (not a vague error)", msg and "secure address" in msg and "8443" in msg, msg)
        await c.shot("v1_insecure_origin")

    async def secure(c):
        await c.call("Page.navigate", url=secure_url + "?theme=dark")
        await asyncio.sleep(2.5)
        check("[B] https origin IS a secure context", await c.js("window.isSecureContext") is True)
        await c.js("document.querySelector('.tab-btn[data-tab=\"voice\"]').click(); true")
        check("no insecure-origin warning on https", await c.js("document.getElementById('voice-secure-note').hidden") is True)
        await c.js("document.getElementById('record-btn').click(); true")
        rec = await c.wait_for("document.getElementById('record-btn').classList.contains('recording')", timeout=10)
        check("microphone granted: button enters recording state", rec)
        await c.shot("v2_recording")
        await asyncio.sleep(5)  # the fake mic plays the spoken question
        await c.js("document.getElementById('record-btn').click(); true")
        t0 = time.time()
        res = await c.wait_for("(()=>{const r=document.getElementById('voice-result');return r.hidden?null:{cls:r.className,text:r.innerText,heard:(r.querySelector('.heard')||{}).innerText||''}})()", timeout=150)
        secs = round(time.time() - t0, 1)
        check("spoken question produced a result", bool(res), str(res)[:200])
        if res:
            print(f"      heard: {res['heard'][:90]!r}\n      answer begins: {res['text'][:110]!r}\n      (text answer {secs}s after stopping)", flush=True)
            check("transcript mentions earthquake", "earthquake" in res["heard"].lower(), res["heard"])
            check("answer came back (not refused)", "answered" in res["cls"], res["cls"])
        audio = await c.wait_for("(()=>{const a=document.querySelector('#voice-result audio');return a?a.src.slice(0,22):null})()", timeout=90)
        check("spoken answer audio is fetched after the text", audio == "data:audio/wav;base64,", str(audio))
        await c.shot("v3_voice_result")
        check("no JavaScript errors", not c.errors, "; ".join(c.errors[:3]))

    await with_browser(wav, insecure)
    await with_browser(wav, secure)
    failed = [r for r in ui_e2e.results if not r[1]]
    print(f"\n{len(ui_e2e.results) - len(failed)}/{len(ui_e2e.results)} voice UI checks passed. Screenshots: {ui_e2e.OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lan-ip", required=True)
    ap.add_argument("--secure-url", default="https://localhost:8443/")
    a = ap.parse_args()
    sys.exit(asyncio.run(main(a.lan_ip, a.secure_url)))
