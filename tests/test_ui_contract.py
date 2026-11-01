"""app.js looks elements up by id; if index.html drops or renames one, the
page breaks silently in the browser (a null.addEventListener at load kills
the whole script). This pins the contract without needing a browser."""

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "disaster_assistant" / "api" / "static"


def _ids_used_by_js():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    return set(re.findall(r'getElementById\("([^"]+)"\)', js))


def test_every_id_app_js_uses_exists_in_index_html():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    missing = [i for i in sorted(_ids_used_by_js()) if f'id="{i}"' not in html]
    assert not missing, f"index.html is missing ids used by app.js: {missing}"


def test_css_files_have_no_external_urls_so_the_ui_works_offline():
    for name in ("style.css", "index.html"):
        text = (STATIC / name).read_text(encoding="utf-8")
        external = re.findall(r'(?:src|href)="(https?://[^"]+)"|url\((https?://[^)]+)\)|@import\s+url\((https?:[^)]+)', text)
        assert not external, f"{name} references external resources: {external}"


def test_all_six_languages_are_in_the_first_launch_picker():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for code in ("en", "hi", "ta", "te", "kn", "ml"):
        assert f'data-lang="{code}"' in html
        # every on-screen string exists in every language
        assert re.search(rf"^\s*{code}: \{{ title:", js, re.M), f"no UI strings for {code}"


def test_the_app_keeps_the_global_hooks_the_phone_test_harness_uses():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for hook in ("window.bridgeAsk", "window.onBridgeResult", "window.onBridgeStage", "window.onBridgePartial", "window.onSpeechDone", "window.onSpeechStep", "window.onMicLevel"):
        assert hook in js
