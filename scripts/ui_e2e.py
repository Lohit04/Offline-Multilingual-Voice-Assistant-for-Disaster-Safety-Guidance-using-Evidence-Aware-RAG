"""End-to-end UI test: a real headless Chrome (DevTools protocol) opens the page
served by the RUNNING server, types questions, clicks buttons and reads what a
user would see. Saves screenshots to experiments/results/ui_e2e/.

Usage: python scripts/ui_e2e.py [--url https://localhost:8443/]
Exit code 0 = every check passed. Needs only `websockets` (already installed).
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
OUT = ROOT / "experiments" / "results" / "ui_e2e"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PORT = 9333


class CDP:
    def __init__(self, ws):
        self.ws, self.n, self.pending, self.errors = ws, 0, {}, []

    async def pump(self):
        async for raw in self.ws:
            m = json.loads(raw)
            if "id" in m and m["id"] in self.pending:
                self.pending.pop(m["id"]).set_result(m)
            elif m.get("method") == "Runtime.exceptionThrown":
                self.errors.append(m["params"]["exceptionDetails"].get("exception", {}).get("description", "exception"))
            elif m.get("method") == "Log.entryAdded":
                e = m["params"]["entry"]
                if e["level"] == "error" and "favicon" not in e.get("url", "") and "ERR_CERT" not in e.get("text", ""):
                    self.errors.append(f"console error: {e['text'][:160]}")

    async def call(self, method, **params):
        self.n += 1
        fut = asyncio.get_running_loop().create_future()
        self.pending[self.n] = fut
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        return (await fut).get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    async def wait_for(self, expr, timeout=90, every=0.4):
        end = time.time() + timeout
        while time.time() < end:
            v = await self.js(expr)
            if v:
                return v
            await asyncio.sleep(every)
        return None

    async def shot(self, name):
        r = await self.call("Page.captureScreenshot", format="png", captureBeyondViewport=True)
        import base64
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{name}.png").write_bytes(base64.b64decode(r["data"]))


results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not ok else ""), flush=True)


RESULT_JS = "(()=>{const r=document.getElementById('text-result');return r.hidden?null:{cls:r.className,text:r.innerText,steps:r.querySelectorAll('.steps li').length,calls:r.querySelectorAll('.call-btn').length,sources:r.querySelectorAll('.source-item').length,understood:!!r.querySelector('.understood'),listen:!!r.querySelector('.listen-btn')}})()"


async def ask(c, question, lang_chip=None):
    if lang_chip:
        await c.js(f"document.querySelector('.lang-chip[data-lang=\"{lang_chip}\"]').click()")
    await c.js(f"document.getElementById('question-input').value = {json.dumps(question)}; document.getElementById('ask-btn').click(); true")
    t0 = time.time()
    res = await c.wait_for(RESULT_JS, timeout=120)
    return res, round(time.time() - t0, 1)


async def main(url):
    profile = tempfile.mkdtemp(prefix="ui_e2e_")
    chrome = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", f"--remote-debugging-port={PORT}",
                               "--ignore-certificate-errors", f"--user-data-dir={profile}", "about:blank"],
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
        assert ws_url, "Chrome DevTools did not start"
        async with websockets.connect(ws_url, max_size=50_000_000) as ws:
            c = CDP(ws)
            pump = asyncio.create_task(c.pump())
            for d in ("Page", "Runtime", "Log"):
                await c.call(f"{d}.enable")
            await c.call("Emulation.setDeviceMetricsOverride", width=390, height=844, deviceScaleFactor=2, mobile=True)
            await c.call("Page.navigate", url=url + ("&" if "?" in url else "?") + "theme=dark")
            await asyncio.sleep(2)

            status = await c.wait_for("(()=>{const t=document.getElementById('status-text').textContent;return t.startsWith('Checking')?null:t})()", timeout=30)
            check("status pill reports Ready", status == "Ready", f"got {status!r}")
            check("emergency dock has 3 numbers", await c.js("document.querySelectorAll('.sos-btn').length") == 3)
            check("6 example chips rendered", await c.js("document.querySelectorAll('#example-chips .chip').length") == 6)
            await c.shot("01_home")

            # English answerable question
            res, secs = await ask(c, "What should I do if floodwater enters my house?", "en")
            check("English flood question answered", res and "answered" in res["cls"], str(res)[:200])
            check("answer rendered as numbered step cards", res and res["steps"] >= 2, f"steps={res and res['steps']}")
            check("sources are shown", res and res["sources"] >= 1)
            check("Listen button offered", res and res["listen"])
            print(f"      (answer took {secs}s)", flush=True)
            await c.shot("02_answer_en")

            # Listen -> audio element appears (async TTS path)
            await c.js("document.querySelector('#text-result .listen-btn').click(); true")
            audio = await c.wait_for("(()=>{const a=document.querySelector('#text-result audio');return a?a.src.slice(0,22):null})()", timeout=60)
            check("Listen fetches audio asynchronously", audio == "data:audio/wav;base64,", str(audio))
            await c.shot("03_listen")

            # Tamil via the example chip, language = auto
            await c.js("document.querySelector('.lang-chip[data-lang=\"auto\"]').click(); document.querySelectorAll('#example-chips .chip')[1].click(); true")
            q = await c.js("document.getElementById('question-input').value")
            check("example chip fills the question box", bool(q) and any("\u0b80" <= ch <= "\u0bff" for ch in q), repr(q))
            await c.js("document.getElementById('ask-btn').click(); true")
            res = await c.wait_for(RESULT_JS, timeout=120)
            check("Tamil question answered", res and "answered" in res["cls"], str(res)[:200])
            check("answer is written in Tamil", res and sum("\u0b80" <= ch <= "\u0bff" for ch in res["text"]) > 20)
            check("'Understood as' shows the English query", res and res["understood"])
            await c.shot("04_answer_ta")

            # Off-topic -> abstain with call buttons
            res, _ = await ask(c, "Who won the cricket match yesterday?", "en")
            check("off-topic question is refused", res and "abstained" in res["cls"], str(res)[:200])
            check("refusal offers 3 call buttons", res and res["calls"] == 3)
            await c.shot("05_refusal")

            # Vague -> clarify, no guessing
            res, _ = await ask(c, "help", "en")
            check("'help' asks for detail instead of guessing", res and "abstained" in res["cls"] and "more detail" in res["text"], str(res)[:200])

            # Theme toggle persists across reload
            await c.js("document.getElementById('theme-toggle').click(); true")
            theme = await c.js("document.documentElement.dataset.theme")
            check("theme toggle switches to light", theme == "light", theme)
            # Reload the PLAIN address: ?theme=... is a preview override that beats the saved choice.
            await c.call("Page.navigate", url=url)
            await asyncio.sleep(2)
            check("theme choice survives reload", await c.js("document.documentElement.dataset.theme") == "light")
            await c.shot("06_light")

            # Voice tab
            await c.js("document.querySelector('.tab-btn[data-tab=\"voice\"]').click(); true")
            check("Speak tab opens", await c.js("document.getElementById('tab-voice').classList.contains('active')"))
            await c.shot("07_voice_tab")

            # Friendly errors
            fe = await c.js("friendlyError('TypeError: Failed to fetch')")
            check("network failure message is actionable", fe and "hotspot" in fe, fe)
            fe2 = await c.js("friendlyError('Local LLM call failed: could not connect to Ollama')")
            check("Ollama failure message tells you what to do", fe2 and "ollama serve" in fe2, fe2)

            check("no JavaScript errors during the whole run", not c.errors, "; ".join(c.errors[:3]))
            pump.cancel()
    finally:
        chrome.terminate()

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} UI checks passed. Screenshots: {OUT}")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://localhost:8443/")
    sys.exit(asyncio.run(main(ap.parse_args().url)))
