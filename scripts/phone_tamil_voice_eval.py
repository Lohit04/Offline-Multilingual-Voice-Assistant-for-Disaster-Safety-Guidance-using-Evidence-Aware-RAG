"""Spoken-Tamil accuracy on the phone: transcribes the 30 answerable Tamil
evaluation questions (TTS-synthesised clips already in the app's cache dir)
with a given whisper model on the phone, then asks the phone's instant
matcher what it would answer. Usage:
    python scripts/phone_tamil_voice_eval.py ggml-base.bin [index.json dir]
Clips are synthetic (single clean voice): an upper bound on real speech."""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.stdout.reconfigure(encoding="utf-8")
import phone_ask, websockets

MODEL = sys.argv[1]
IDX = json.loads(Path(sys.argv[2], "index.json").read_text(encoding="utf-8"))

async def main():
    async with websockets.connect(phone_ask.connect_url(), max_size=2**24, ping_interval=None) as ws:
        n = [0]
        async def ev(expr, aw=True):
            for _ in range(3):
                n[0] += 1
                await ws.send(json.dumps({"id": n[0], "method": "Runtime.evaluate",
                                          "params": {"expression": expr, "awaitPromise": aw, "returnByValue": True}}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == n[0]: break
                v = m.get("result", {}).get("result", {}).get("value")
                if v is not None: return v
                await asyncio.sleep(2)
            return None
        while await ev("typeof AndroidBridge") != "object": await asyncio.sleep(2)
        rows, times = [], []
        for it in IDX:
            d = json.loads(await ev("callBridge('debugTranscribe2',[%s,%s,'ta',4]).then(d=>JSON.stringify(d))"
                                    % (json.dumps(it["file"]), json.dumps(MODEL))) or "{}")
            t = (d.get("text") or "").strip(); times.append(d.get("ms", 0) / 1000)
            mid = await ev("AndroidBridge.instantMatchId(%s)" % json.dumps(t), False) or ""
            haz = mid.split(".")[0] if mid else ""
            ok = bool(mid) and (haz in it["cats"] or (haz == "urban_flood" and "flood" in it["cats"]))
            rows.append({**it, "transcript": t, "match": mid, "ok": ok})
            print(f"{it['id']} {times[-1]:.1f}s -> {mid or '-':24} {'OK' if ok else ('WRONG' if mid else '')} {t[:40]!r}", flush=True)
        ts = sorted(times)
        print(f"\n{MODEL}: instant & right hazard {sum(r['ok'] for r in rows)}/30, wrong hazard "
              f"{sum(1 for r in rows if r['match'] and not r['ok'])}; time median {ts[15]:.1f}s max {ts[-1]:.1f}s")
        Path("experiments/results", f"phone_tamil_voice_{MODEL.replace('.bin','')}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

asyncio.run(main())
