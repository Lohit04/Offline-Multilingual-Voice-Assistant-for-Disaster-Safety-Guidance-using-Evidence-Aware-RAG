// Disaster Safety Assistant -- app-style UI (2026-10-27 redesign).
//
// One script for both places it runs:
//  - the web page served by api/server.py (fetch() to /api/...), and
//  - the Android app (bundled in assets/www, talking to Java through the
//    AndroidBridge object that MainActivity injects: answers, speech and
//    read-aloud all happen on the phone, with no network).
// Design rule: a frightened person should not have to read or decide much.
// Home = one big mic + picture tiles; an answer = big numbered steps, read
// aloud; the red call bar never moves.

"use strict";

const ON_DEVICE = typeof AndroidBridge !== "undefined";
const BUILD_TAG = "build 4.8 (2026-10-27)";
const LANGS = ["en", "hi", "ta", "te", "kn", "ml"];
const NATIVE_NAME = { en: "English", hi: "हिन्दी", ta: "தமிழ்", te: "తెలుగు", kn: "ಕನ್ನಡ", ml: "മലയാളം" };

// Short UI wording. Non-English text written by the AI assistant and NOT
// reviewed by native speakers -- kept deliberately short and plain.
const STR = {
  en: { title: "Disaster Help", micHint: "Tap and say what's happening", listening: "Listening… speak now",
        listeningWeb: "Listening… tap again when done", finding: "Finding the steps…", typeInstead: "Type instead",
        placeholder: "e.g. water is entering my house", tilesTitle: "Or tap what's happening",
        listen: "Listen", stop: "Stop", askAnother: "Ask another question", call112: "Call 112",
        before: "Before", during: "During", after: "After", instant: "Official guidance", grounded: "From official guidance",
        cantConfirm: "Can't confirm", source: "Where this comes from", noVoice: "Voice isn't available offline in this language on this phone. Tap a picture below or type.",
        notHeard: "I didn't catch that. Tap and try again.", noMic: "Microphone is blocked. Allow it, or type instead.",
        fineprint: "Official NDMA guidance · works without internet · not a substitute for emergency services",
        offline: "Works offline", ready: "Ready", noServer: "Can't reach the assistant", error: "Something went wrong. Please try again." },
  hi: { title: "आपदा सहायता", micHint: "टैप करें और बताएं क्या हो रहा है", listening: "सुन रहा हूँ… बोलिए",
        listeningWeb: "सुन रहा हूँ… बोलने के बाद फिर टैप करें", finding: "कदम ढूँढ रहे हैं…", typeInstead: "लिखकर पूछें",
        placeholder: "जैसे: घर में पानी घुस रहा है", tilesTitle: "या चुनें क्या हो रहा है",
        listen: "सुनें", stop: "रोकें", askAnother: "दूसरा सवाल पूछें", call112: "112 पर कॉल करें",
        before: "पहले", during: "दौरान", after: "बाद में", instant: "आधिकारिक मार्गदर्शन", grounded: "आधिकारिक मार्गदर्शन से",
        cantConfirm: "पुष्टि नहीं कर सकते", source: "यह जानकारी कहाँ से है", noVoice: "इस फ़ोन पर इस भाषा में आवाज़ ऑफ़लाइन उपलब्ध नहीं है। नीचे कोई चित्र चुनें या लिखें।",
        notHeard: "समझ नहीं आया। फिर से टैप करें।", noMic: "माइक्रोफ़ोन बंद है। अनुमति दें या लिखें।",
        fineprint: "NDMA का आधिकारिक मार्गदर्शन · बिना इंटरनेट के चलता है · आपातकालीन सेवाओं का विकल्प नहीं",
        offline: "ऑफ़लाइन चलता है", ready: "तैयार", noServer: "सहायक से संपर्क नहीं हो पा रहा", error: "कुछ गड़बड़ हुई। फिर से कोशिश करें।" },
  ta: { title: "பேரிடர் உதவி", micHint: "தட்டி, என்ன நடக்கிறது என்று சொல்லுங்கள்", listening: "கேட்கிறேன்… பேசுங்கள்",
        listeningWeb: "கேட்கிறேன்… முடிந்ததும் மீண்டும் தட்டுங்கள்", finding: "வழிமுறைகளைத் தேடுகிறது…", typeInstead: "தட்டச்சு செய்யுங்கள்",
        placeholder: "உ.தா: வீட்டுக்குள் தண்ணீர் வருகிறது", tilesTitle: "அல்லது என்ன நடக்கிறது என்பதைத் தட்டுங்கள்",
        listen: "கேளுங்கள்", stop: "நிறுத்து", askAnother: "வேறு கேள்வி கேளுங்கள்", call112: "112 அழைக்கவும்",
        before: "முன்", during: "போது", after: "பின்", instant: "அதிகாரப்பூர்வ வழிகாட்டுதல்", grounded: "அதிகாரப்பூர்வ வழிகாட்டுதலில் இருந்து",
        cantConfirm: "உறுதிப்படுத்த முடியவில்லை", source: "இந்தத் தகவல் எங்கிருந்து", noVoice: "இந்த போனில் இந்த மொழியில் குரல் வசதி ஆஃப்லைனில் இல்லை. கீழே ஒரு படத்தைத் தட்டுங்கள் அல்லது தட்டச்சு செய்யுங்கள்.",
        notHeard: "புரியவில்லை. மீண்டும் தட்டுங்கள்.", noMic: "மைக் அனுமதி இல்லை. அனுமதிக்கவும் அல்லது தட்டச்சு செய்யவும்.",
        fineprint: "NDMA அதிகாரப்பூர்வ வழிகாட்டுதல் · இணையம் இல்லாமல் இயங்கும் · அவசர சேவைகளுக்கு மாற்று அல்ல",
        offline: "ஆஃப்லைனில் இயங்கும்", ready: "தயார்", noServer: "உதவியாளரை அணுக முடியவில்லை", error: "ஏதோ தவறு நடந்தது. மீண்டும் முயற்சிக்கவும்." },
  te: { title: "విపత్తు సహాయం", micHint: "నొక్కి, ఏమి జరుగుతోందో చెప్పండి", listening: "వింటున్నాను… మాట్లాడండి",
        listeningWeb: "వింటున్నాను… పూర్తయ్యాక మళ్ళీ నొక్కండి", finding: "సూచనలు వెతుకుతోంది…", typeInstead: "టైప్ చేయండి",
        placeholder: "ఉదా: ఇంట్లోకి నీళ్లు వస్తున్నాయి", tilesTitle: "లేదా ఏమి జరుగుతోందో ఎంచుకోండి",
        listen: "వినండి", stop: "ఆపు", askAnother: "మరో ప్రశ్న అడగండి", call112: "112కి కాల్ చేయండి",
        before: "ముందు", during: "సమయంలో", after: "తర్వాత", instant: "అధికారిక మార్గదర్శకం", grounded: "అధికారిక మార్గదర్శకం నుండి",
        cantConfirm: "నిర్ధారించలేము", source: "ఈ సమాచారం ఎక్కడి నుండి", noVoice: "ఈ ఫోన్‌లో ఈ భాషలో వాయిస్ ఆఫ్‌లైన్‌లో లేదు. కింద ఒక బొమ్మను నొక్కండి లేదా టైప్ చేయండి.",
        notHeard: "అర్థం కాలేదు. మళ్ళీ నొక్కండి.", noMic: "మైక్ అనుమతి లేదు. అనుమతించండి లేదా టైప్ చేయండి.",
        fineprint: "NDMA అధికారిక మార్గదర్శకం · ఇంటర్నెట్ లేకుండా పనిచేస్తుంది · అత్యవసర సేవలకు ప్రత్యామ్నాయం కాదు",
        offline: "ఆఫ్‌లైన్‌లో పనిచేస్తుంది", ready: "సిద్ధం", noServer: "సహాయకుడిని చేరుకోలేకపోయాం", error: "ఏదో తప్పు జరిగింది. మళ్ళీ ప్రయత్నించండి." },
  kn: { title: "ವಿಪತ್ತು ಸಹಾಯ", micHint: "ಒತ್ತಿ, ಏನಾಗುತ್ತಿದೆ ಎಂದು ಹೇಳಿ", listening: "ಕೇಳುತ್ತಿದ್ದೇನೆ… ಮಾತನಾಡಿ",
        listeningWeb: "ಕೇಳುತ್ತಿದ್ದೇನೆ… ಮುಗಿದ ನಂತರ ಮತ್ತೆ ಒತ್ತಿ", finding: "ಕ್ರಮಗಳನ್ನು ಹುಡುಕುತ್ತಿದೆ…", typeInstead: "ಟೈಪ್ ಮಾಡಿ",
        placeholder: "ಉದಾ: ಮನೆಗೆ ನೀರು ನುಗ್ಗುತ್ತಿದೆ", tilesTitle: "ಅಥವಾ ಏನಾಗುತ್ತಿದೆ ಎಂಬುದನ್ನು ಆಯ್ಕೆಮಾಡಿ",
        listen: "ಕೇಳಿ", stop: "ನಿಲ್ಲಿಸಿ", askAnother: "ಇನ್ನೊಂದು ಪ್ರಶ್ನೆ ಕೇಳಿ", call112: "112 ಗೆ ಕರೆ ಮಾಡಿ",
        before: "ಮೊದಲು", during: "ಸಮಯದಲ್ಲಿ", after: "ನಂತರ", instant: "ಅಧಿಕೃತ ಮಾರ್ಗದರ್ಶನ", grounded: "ಅಧಿಕೃತ ಮಾರ್ಗದರ್ಶನದಿಂದ",
        cantConfirm: "ದೃಢೀಕರಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ", source: "ಈ ಮಾಹಿತಿ ಎಲ್ಲಿಂದ", noVoice: "ಈ ಫೋನ್‌ನಲ್ಲಿ ಈ ಭಾಷೆಯಲ್ಲಿ ಧ್ವನಿ ಆಫ್‌ಲೈನ್‌ನಲ್ಲಿ ಲಭ್ಯವಿಲ್ಲ. ಕೆಳಗೆ ಒಂದು ಚಿತ್ರವನ್ನು ಒತ್ತಿ ಅಥವಾ ಟೈಪ್ ಮಾಡಿ.",
        notHeard: "ಅರ್ಥವಾಗಲಿಲ್ಲ. ಮತ್ತೆ ಒತ್ತಿ.", noMic: "ಮೈಕ್ ಅನುಮತಿ ಇಲ್ಲ. ಅನುಮತಿಸಿ ಅಥವಾ ಟೈಪ್ ಮಾಡಿ.",
        fineprint: "NDMA ಅಧಿಕೃತ ಮಾರ್ಗದರ್ಶನ · ಇಂಟರ್ನೆಟ್ ಇಲ್ಲದೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ · ತುರ್ತು ಸೇವೆಗಳಿಗೆ ಪರ್ಯಾಯವಲ್ಲ",
        offline: "ಆಫ್‌ಲೈನ್‌ನಲ್ಲಿ ಕೆಲಸ ಮಾಡುತ್ತದೆ", ready: "ಸಿದ್ಧ", noServer: "ಸಹಾಯಕನನ್ನು ತಲುಪಲು ಸಾಧ್ಯವಾಗಿಲ್ಲ", error: "ಏನೋ ತಪ್ಪಾಗಿದೆ. ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ." },
  ml: { title: "ദുരന്ത സഹായം", micHint: "അമർത്തി, എന്താണ് സംഭവിക്കുന്നതെന്ന് പറയുക", listening: "കേൾക്കുന്നു… സംസാരിക്കൂ",
        listeningWeb: "കേൾക്കുന്നു… കഴിഞ്ഞാൽ വീണ്ടും അമർത്തുക", finding: "നിർദ്ദേശങ്ങൾ തിരയുന്നു…", typeInstead: "ടൈപ്പ് ചെയ്യുക",
        placeholder: "ഉദാ: വീട്ടിൽ വെള്ളം കയറുന്നു", tilesTitle: "അല്ലെങ്കിൽ എന്താണ് സംഭവിക്കുന്നതെന്ന് തിരഞ്ഞെടുക്കുക",
        listen: "കേൾക്കുക", stop: "നിർത്തുക", askAnother: "മറ്റൊരു ചോദ്യം ചോദിക്കുക", call112: "112 വിളിക്കുക",
        before: "മുമ്പ്", during: "സമയത്ത്", after: "ശേഷം", instant: "ഔദ്യോഗിക മാർഗ്ഗനിർദ്ദേശം", grounded: "ഔദ്യോഗിക മാർഗ്ഗനിർദ്ദേശത്തിൽ നിന്ന്",
        cantConfirm: "സ്ഥിരീകരിക്കാനാവില്ല", source: "ഈ വിവരം എവിടെ നിന്ന്", noVoice: "ഈ ഫോണിൽ ഈ ഭാഷയിൽ ശബ്ദം ഓഫ്‌ലൈനായി ലഭ്യമല്ല. താഴെ ഒരു ചിത്രം അമർത്തുക അല്ലെങ്കിൽ ടൈപ്പ് ചെയ്യുക.",
        notHeard: "മനസ്സിലായില്ല. വീണ്ടും അമർത്തുക.", noMic: "മൈക്ക് അനുമതിയില്ല. അനുവദിക്കുക അല്ലെങ്കിൽ ടൈപ്പ് ചെയ്യുക.",
        fineprint: "NDMA ഔദ്യോഗിക മാർഗ്ഗനിർദ്ദേശം · ഇന്റർനെറ്റ് ഇല്ലാതെ പ്രവർത്തിക്കും · അടിയന്തര സേവനങ്ങൾക്ക് പകരമല്ല",
        offline: "ഓഫ്‌ലൈനായി പ്രവർത്തിക്കും", ready: "തയ്യാർ", noServer: "സഹായിയെ ബന്ധപ്പെടാനാകുന്നില്ല", error: "എന്തോ പിഴവ് സംഭവിച്ചു. വീണ്ടും ശ്രമിക്കുക." },
};

// Picture tiles: the prepared situations' hazards (scripts/build_answer_cache.py).
// Names are the same hazard words as translation.py's glossary.
const HAZARDS = [
  { id: "flood", icon: "🌧️", name: { en: "Flood", hi: "बाढ़", ta: "வெள்ளம்", te: "వరద", kn: "ಪ್ರವಾಹ", ml: "വെള്ളപ്പൊക്കം" } },
  { id: "cyclone", icon: "🌀", name: { en: "Cyclone", hi: "चक्रवात", ta: "புயல்", te: "తుఫాను", kn: "ಚಂಡಮಾರುತ", ml: "ചുഴലിക്കാറ്റ്" } },
  { id: "earthquake", icon: "🏚️", name: { en: "Earthquake", hi: "भूकंप", ta: "நிலநடுக்கம்", te: "భూకంపం", kn: "ಭೂಕಂಪ", ml: "ഭൂകമ്പം" } },
  { id: "fire", icon: "🔥", name: { en: "Fire", hi: "आग", ta: "தீ விபத்து", te: "అగ్ని ప్రమాదం", kn: "ಬೆಂಕಿ", ml: "തീപിടിത്തം" } },
  { id: "heat_wave", icon: "🌡️", name: { en: "Heat wave", hi: "लू", ta: "வெப்ப அலை", te: "వడగాలి", kn: "ಬಿಸಿಗಾಳಿ", ml: "ഉഷ്ണതരംഗം" } },
  { id: "landslide", icon: "⛰️", name: { en: "Landslide", hi: "भूस्खलन", ta: "நிலச்சரிவு", te: "కొండచరియలు", kn: "ಭೂಕುಸಿತ", ml: "മണ്ണിടിച്ചിൽ" } },
  { id: "tsunami", icon: "🌊", name: { en: "Tsunami", hi: "सुनामी", ta: "சுனாமி", te: "సునామీ", kn: "ಸುನಾಮಿ", ml: "സുനാമി" } },
  { id: "lightning", icon: "⚡", name: { en: "Lightning", hi: "आकाशीय बिजली", ta: "மின்னல்", te: "పిడుగు", kn: "ಸಿಡಿಲು", ml: "ഇടിമിന്നൽ" } },
];
const PHASES = ["before", "during", "after"];

const DOC_NAMES = {
  ndma_floods_2008: "NDMA Flood Guidelines (2008)",
  ndma_earthquakes_2007: "NDMA Earthquake Guidelines (2007)",
  ndma_cyclones_2008: "NDMA Cyclone Guidelines (2008)",
  ndma_disaster_survival_guide_2020: "NDMA Disaster Survival Guide (2020)",
};
const ICON_OK = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 4 5v6c0 5 3.4 9.4 8 11 4.6-1.6 8-6 8-11V5l-8-3zm-1.2 14.2-3.5-3.5 1.4-1.4 2.1 2.1 4.6-4.6 1.4 1.4-6 6z" fill="currentColor"/></svg>';
const ICON_WARN = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 1 21h22L12 2zm1 15h-2v-2h2v2zm0-4h-2V9h2v4z" fill="currentColor"/></svg>';

const $ = (id) => document.getElementById(id);
let lang = null;
let instantIds = new Set();

function t(key) { return (STR[lang] || STR.en)[key] || STR.en[key]; }
function makeEl(tag, className, text) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text !== undefined) e.textContent = text;
  return e;
}

// ---------- instant feedback: haptics + ripples ----------
// Phone: the view's native haptic tick (no permission; follows the user's
// touch-vibration setting). Web: the Vibration API where it exists.
const VIBRATE = { tap: 10, start: 25, success: [12, 60, 18], error: [40, 50, 40] };
function haptic(kind) {
  try {
    if (ON_DEVICE && AndroidBridge.haptic) AndroidBridge.haptic(kind);
    else if (navigator.vibrate) navigator.vibrate(VIBRATE[kind] || 10);
  } catch (e) {}
}
const RIPPLE = ".tile, .lang-chip, .listen-btn, .again-btn, .link-btn, .phase-tabs button, .sos-bar a, .call-row a, .icon-btn, .lang-btn, .send";
document.addEventListener("pointerdown", (e) => {
  const b = e.target.closest(RIPPLE);
  if (!b || b.disabled) return;
  const r = b.getBoundingClientRect();
  const size = Math.max(r.width, r.height) * 2.2;
  const s = makeEl("span", "ripple");
  s.style.width = s.style.height = size + "px";
  s.style.left = (e.clientX - r.left - size / 2) + "px";
  s.style.top = (e.clientY - r.top - size / 2) + "px";
  b.appendChild(s);
  setTimeout(() => s.remove(), 600);
}, { passive: true });
function shakeHint(text) {
  const h = $("mic-hint");
  setHint(text);
  h.classList.remove("shake");
  void h.offsetWidth;  // restart the animation
  h.classList.add("shake");
  haptic("error");
}

// ---------- Android bridge plumbing (phone app only) ----------
const _cb = {};
const _progress = {};
window.onBridgeResult = function (id, json) {
  const cb = _cb[id];
  if (!cb) return;
  delete _cb[id];
  delete _progress[id];
  try { cb(JSON.parse(json)); } catch (e) { cb({ error: "bad result: " + e.message }); }
};
window.onBridgeStage = function (id, label) { if (_progress[id]) _progress[id]({ stage: label }); };
window.onBridgePartial = function (id, text) { if (_progress[id]) _progress[id]({ text: text }); };
function callBridge(method, args, onProgress) {
  return new Promise((resolve) => {
    const id = "r" + Math.random().toString(36).slice(2);
    _cb[id] = resolve;
    if (onProgress) _progress[id] = onProgress;
    AndroidBridge[method](...args, id);
  });
}
// Used by scripts/phone_ask.py over Chrome DevTools to time on-device answers.
window.bridgeAsk = (question, language) => callBridge("askQuestion", [question, language || "auto"]);

// ---------- language ----------
function savedLang() { try { return localStorage.getItem("lang"); } catch (e) { return null; } }
function setLang(code) {
  lang = code;
  try { localStorage.setItem("lang", code); } catch (e) {}
  document.documentElement.lang = code;
  document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.dataset.i18n); });
  $("question-input").placeholder = t("placeholder");
  $("lang-btn").textContent = NATIVE_NAME[code];
  document.querySelectorAll(".lang-chip").forEach((c) => {
    const on = c.dataset.lang === code;
    c.classList.toggle("active", on);
    c.setAttribute("aria-checked", on ? "true" : "false");
  });
  renderTiles();
  updateMic();
  setStatus();
  // Phone: load this language's voice now so the first answer speaks at once.
  if (ON_DEVICE && AndroidBridge.prepareVoice) AndroidBridge.prepareVoice(code);
}
document.querySelectorAll(".lang-chip").forEach((c) => c.addEventListener("click", () => {
  haptic("tap");
  setLang(c.dataset.lang);
  setTimeout(() => showHome(true), 140);  // let the chip's press show first
}));
$("lang-btn").addEventListener("click", () => { haptic("tap"); $("home").hidden = true; $("lang-screen").hidden = false; });

// ---------- screens ----------
function showHome(entrance) {
  $("lang-screen").hidden = true;
  $("answer-screen").hidden = true;
  const home = $("home");
  home.classList.toggle("enter", !!entrance);
  home.hidden = false;
  stopSpeaking();
}
function openAnswer(title) {
  $("home").hidden = true;
  $("answer-screen").hidden = false;
  $("answer-title").textContent = title;
  $("phase-tabs").hidden = true;
  $("listen-btn").hidden = true;
  $("answer-source").hidden = true;
  window.scrollTo(0, 0);
  // Phone/browser Back closes the answer instead of leaving the app.
  if (!history.state || !history.state.answer) history.pushState({ answer: true }, "");
  $("answer-title").focus();
}
window.addEventListener("popstate", () => { if (!$("answer-screen").hidden) showHome(); });
$("back-btn").addEventListener("click", () => { haptic("tap"); history.back(); });
$("again-btn").addEventListener("click", () => { haptic("tap"); history.back(); });

function showLoading(stage) {
  const body = $("answer-body");
  body.innerHTML = "";
  const sk = makeEl("div", "skeleton");
  for (let i = 0; i < 3; i++) sk.appendChild(makeEl("span"));
  body.appendChild(sk);
  body.appendChild(makeEl("p", "stage", stage || t("finding")));
}
function setStage(stage) {
  const p = $("answer-body").querySelector(".stage");
  if (p) p.textContent = stage;
}
function showStreaming(text) {
  let el = $("answer-body").querySelector(".streaming");
  if (!el) {
    $("answer-body").innerHTML = "";
    el = makeEl("p", "streaming");
    $("answer-body").appendChild(el);
  }
  el.textContent = text;
}

// ---------- rendering an answer ----------
function appendInline(parent, text) {
  text.split(/(\*\*[^*]+\*\*)/).forEach((part) => {
    if (/^\*\*[^*]+\*\*$/.test(part)) parent.appendChild(makeEl("strong", "", part.slice(2, -2)));
    else if (part) parent.appendChild(document.createTextNode(part));
  });
}
// Renders the answer as step cards and returns the read-aloud segments:
// one per card/paragraph, so the card being spoken can be lit up.
function renderSteps(el, text) {
  const segments = [];
  let list = null;
  let listType = null;
  let n = 0;
  text.split("\n").forEach((raw) => {
    const line = raw.trim();
    if (!line) { list = null; listType = null; return; }
    const num = line.match(/^(\d+)[.)]\s+(.*)$/);
    const bul = line.match(/^[*\-•]\s+(.*)$/);
    let node;
    let spoken;
    if (num || bul) {
      const type = num ? "ol" : "ul";
      if (!list || listType !== type) {
        list = makeEl(type, num ? "steps" : "bullets");
        listType = type;
        el.appendChild(list);
      }
      node = document.createElement("li");
      node.style.setProperty("--i", n++);
      spoken = num ? num[2] : bul[1];
      appendInline(node, spoken);
      list.appendChild(node);
    } else {
      list = null; listType = null;
      node = document.createElement("p");
      spoken = line;
      appendInline(node, line);
      el.appendChild(node);
    }
    segments.push({ el: node, text: spoken.replace(/\*\*/g, "") });
  });
  return segments;
}

let current = null;  // the answer on screen: { text, language, segments }

// Response time shown on every answer: from the tap (or the end of speech)
// to the answer on screen -- the latency the person actually waits.
let askT0 = null;
function startTimer() { askT0 = performance.now(); }
function fmtSecs(s) { return (s < 1 ? s.toFixed(2) : s.toFixed(1)) + " s"; }

function renderAnswer(data, { speakNow, secs } = {}) {
  if (secs === undefined && askT0 !== null) secs = (performance.now() - askT0) / 1000;
  askT0 = null;
  const body = $("answer-body");
  body.innerHTML = "";
  if (data.error) {
    body.appendChild(makeEl("p", "abstain", data.empty_transcription ? t("notHeard") : t("error")));
    current = null;
    haptic("error");
    return;
  }
  const badge = makeEl("span", "badge " + (data.abstained ? "warn" : data.cached ? "instant" : "ok"));
  badge.innerHTML = data.abstained ? ICON_WARN : ICON_OK;
  badge.appendChild(document.createTextNode(
    data.abstained ? t("cantConfirm") : (data.cached ? "⚡ " + t("instant") : t("grounded"))));
  const row = makeEl("div", "badge-row");
  row.appendChild(badge);
  if (typeof secs === "number" && isFinite(secs)) {
    const tm = makeEl("span", "timing", "⏱ " + fmtSecs(secs));
    tm.title = "Time from your question to this answer";
    row.appendChild(tm);
  }
  body.appendChild(row);

  let segments;
  if (data.abstained) {
    const box = makeEl("div", "abstain");
    appendInline(box, data.text || "");
    body.appendChild(box);
    segments = [{ el: box, text: (data.text || "").replace(/\*\*/g, "") }];
    const row = makeEl("div", "call-row");
    [["112", "main"], ["1070", ""], ["1077", ""]].forEach(([n, cls]) => {
      const a = makeEl("a", cls, n === "112" ? t("call112") : n);
      a.href = "tel:" + n;
      row.appendChild(a);
    });
    body.appendChild(row);
  } else {
    segments = renderSteps(body, data.text || "");
  }
  haptic(data.abstained ? "tap" : "success");

  const src = $("answer-source");
  src.innerHTML = "";
  if (data.sources && data.sources.length) {
    src.appendChild(makeEl("summary", "", t("source")));
    const seen = new Set();
    data.sources.forEach((s) => {
      const key = s.document_id + ":" + s.page;
      if (seen.has(key)) return;
      seen.add(key);
      const item = makeEl("div", "src");
      item.appendChild(makeEl("b", "", (DOC_NAMES[s.document_id] || s.document_id) + " · p. " + s.page));
      item.appendChild(document.createTextNode((s.text || "").slice(0, 240) + ((s.text || "").length > 240 ? "…" : "")));
      src.appendChild(item);
    });
    src.hidden = false;
  }

  current = { text: data.text || "", language: data.answer_language || lang, segments: segments };
  setListenButton(false);
  $("listen-btn").hidden = !current.text;
  if (speakNow && current.text) speak();
}

// ---------- read aloud, lighting up the step being spoken ----------
let audio = null;
let speaking = false;
function setListenButton(playing) {
  speaking = playing;
  $("listen-btn").classList.toggle("playing", playing);
  $("listen-label").textContent = playing ? t("stop") : t("listen");
}
let spokenIndex = -1;
function highlightStep(i) {
  if (!current || !current.segments || i === spokenIndex) return;
  spokenIndex = i;
  current.segments.forEach((s, k) => {
    s.el.classList.toggle("speaking", k === i);
    s.el.classList.toggle("spoken", k < i);
  });
  const el = current.segments[i] && current.segments[i].el;
  if (!el) return;
  const r = el.getBoundingClientRect();
  // Keep the step being read on screen (above the call bar).
  if (r.top < 80 || r.bottom > window.innerHeight - 100) el.scrollIntoView({ behavior: "smooth", block: "center" });
}
function clearHighlight() {
  spokenIndex = -1;
  if (current && current.segments) current.segments.forEach((s) => s.el.classList.remove("speaking", "spoken"));
}
window.onSpeechStep = function (i) { if (speaking) highlightStep(i); };
window.onSpeechDone = function () { setListenButton(false); clearHighlight(); };

async function speak() {
  if (!current) return;
  const parts = current.segments.map((s) => s.text).filter(Boolean);
  if (ON_DEVICE) {
    const ok = AndroidBridge.speakSteps
      ? AndroidBridge.speakSteps(JSON.stringify(parts), current.language)
      : AndroidBridge.speakText(current.text, current.language);
    if (ok) setListenButton(true);
    return;
  }
  setListenButton(true);
  try {
    const form = new FormData();
    form.append("text", current.text);
    form.append("language", current.language);
    const j = await (await fetch("/api/tts", { method: "POST", body: form })).json();
    if (!j.audio_base64 || !speaking) { setListenButton(false); return; }
    audio = new Audio("data:audio/wav;base64," + j.audio_base64);
    // One audio file for the whole answer: estimate which step is playing
    // from its share of the text (close enough to follow along).
    const lens = parts.map((p) => p.length);
    const total = lens.reduce((a, b) => a + b, 0) || 1;
    audio.ontimeupdate = () => {
      if (!audio || !audio.duration) return;
      let at = (audio.currentTime / audio.duration) * total;
      let i = 0;
      while (i < lens.length - 1 && at > lens[i]) { at -= lens[i]; i++; }
      highlightStep(i);
    };
    audio.onended = () => { setListenButton(false); clearHighlight(); };
    await audio.play();
  } catch (e) { setListenButton(false); }
}
function stopSpeaking() {
  if (ON_DEVICE) { try { AndroidBridge.stopSpeaking(); } catch (e) {} }
  if (audio) { audio.pause(); audio = null; }
  setListenButton(false);
  clearHighlight();
}
$("listen-btn").addEventListener("click", () => { haptic("tap"); speaking ? stopSpeaking() : speak(); });

// ---------- picture tiles: one tap -> prepared official steps ----------
async function loadInstantIds() {
  try {
    const ids = ON_DEVICE ? JSON.parse(AndroidBridge.instantIds())
      : (await (await fetch("/api/instant_ids")).json()).ids;
    instantIds = new Set(ids);
  } catch (e) { instantIds = new Set(); }
  renderTiles();
}
function renderTiles() {
  const box = $("example-chips");
  box.innerHTML = "";
  let i = 0;
  HAZARDS.forEach((h) => {
    if (!PHASES.some((p) => instantIds.has(h.id + "." + p))) return;
    const b = makeEl("button", "tile");
    b.type = "button";
    b.style.setProperty("--i", i++);
    b.appendChild(makeEl("span", "emoji", h.icon));
    const name = h.name[lang] || h.name.en;
    if (name.length > 8 && !name.includes(" ")) b.classList.add("long");
    b.appendChild(makeEl("span", "name", name));
    b.addEventListener("click", () => {
      haptic("tap");
      openSituation(h, instantIds.has(h.id + ".during") ? "during" : PHASES.find((p) => instantIds.has(h.id + "." + p)));
    });
    box.appendChild(b);
  });
}
function renderTabs(h, phase) {
  const tabs = $("phase-tabs");
  if (tabs.dataset.hazard !== h.id) {
    tabs.dataset.hazard = h.id;
    tabs.innerHTML = "";
    tabs.appendChild(makeEl("span", "pill"));
    PHASES.forEach((p) => {
      const b = makeEl("button", "", t(p));
      b.type = "button";
      b.dataset.phase = p;
      b.setAttribute("role", "tab");
      b.disabled = !instantIds.has(h.id + "." + p);
      b.addEventListener("click", () => {
        if (b.getAttribute("aria-selected") === "true") return;
        haptic("tap");
        stopSpeaking();
        openSituation(h, p);
      });
      tabs.appendChild(b);
    });
  }
  const idx = PHASES.indexOf(phase);
  tabs.querySelectorAll("button").forEach((b) => b.setAttribute("aria-selected", b.dataset.phase === phase ? "true" : "false"));
  tabs.querySelector(".pill").style.transform = "translateX(calc(" + idx + " * (100% + 6px)))";
  tabs.hidden = false;
}
async function openSituation(h, phase) {
  startTimer();
  if ($("answer-screen").hidden) {
    openAnswer(h.icon + " " + (h.name[lang] || h.name.en));
    $("phase-tabs").dataset.hazard = "";
  }
  renderTabs(h, phase);
  let data;
  try {
    data = ON_DEVICE ? JSON.parse(AndroidBridge.instantById(h.id + "." + phase, lang))
      : await (await fetch("/api/instant?id=" + encodeURIComponent(h.id + "." + phase) + "&language=" + lang)).json();
  } catch (e) { data = { error: String(e) }; }
  renderAnswer(data, { speakNow: true });
}

// ---------- typed question ----------
$("type-btn").addEventListener("click", () => {
  haptic("tap");
  $("type-form").hidden = false;
  $("type-btn").hidden = true;
  $("question-input").focus();
});
$("type-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("question-input").value.trim();
  if (!q) { shakeHint(t("micHint")); return; }
  haptic("tap");
  $("question-input").blur();
  startTimer();
  await ask(q, false);
});
async function ask(question, spoken) {
  openAnswer("“" + question + "”");
  showLoading();
  let data;
  try {
    if (ON_DEVICE) {
      data = await callBridge("askQuestion", [question, "auto"], (p) => {
        if (p.stage) setStage(p.stage);
        if (p.text) showStreaming(p.text);
      });
    } else {
      const form = new FormData();
      form.append("question", question);
      form.append("language", "auto");
      data = await (await fetch("/api/ask", { method: "POST", body: form })).json();
    }
  } catch (e) { data = { error: String(e) }; }
  renderAnswer(data, { speakNow: spoken });
}

// ---------- voice ----------
// Phone: Android's on-device recognizer (English, Hindi) streams words and
// stops by itself; Tamil/Telugu/Kannada/Malayalam have no usable offline
// engine on the test phone (docs/implementation_status.md, Session 13), so
// the mic points people to the tiles and typing instead.
// Web: record, then the server transcribes with the chosen language.
// Both: the ring around the mic grows with the speaker's loudness, so it is
// obvious at a glance that the app is hearing them.
let listening = false;
let recorder = null;
let chunks = [];

function phoneVoiceOk() {
  return ON_DEVICE && AndroidBridge.nativeSpeechSupports && AndroidBridge.nativeSpeechSupports(lang);
}
function webVoiceOk() {
  return !ON_DEVICE && window.isSecureContext && navigator.mediaDevices && window.MediaRecorder;
}
function updateMic() {
  const ok = ON_DEVICE ? phoneVoiceOk() : webVoiceOk();
  $("mic-btn").classList.toggle("unavailable", !ok);
  $("mic-btn").setAttribute("aria-label", t("micHint"));
  setHint(ok ? t("micHint") : (ON_DEVICE ? t("noVoice") : t("noMic")));
}
function setHint(text, heard) {
  $("mic-hint").textContent = text;
  $("mic-hint").classList.toggle("heard", !!heard);
  $("mic-hint").classList.remove("shake");
}
function setMicState(state) {  // "idle" | "starting" | "listening"
  const mic = $("mic-btn");
  mic.classList.toggle("starting", state === "starting");
  mic.classList.toggle("listening", state === "listening");
  listening = state !== "idle";
  if (state !== "listening") mic.style.setProperty("--level", 0);
}
window.onMicLevel = function (level) { $("mic-btn").style.setProperty("--level", level); };

$("mic-btn").addEventListener("click", async () => {
  if (ON_DEVICE) return phoneVoice();
  return webVoice();
});

async function phoneVoice() {
  if (!phoneVoiceOk()) { shakeHint(t("noVoice")); return; }
  if (listening) { haptic("tap"); AndroidBridge.stopListening(); return; }
  haptic("tap");
  stopSpeaking();
  setMicState("starting");  // visible the instant the finger lands
  setHint(t("listening"));
  let heard = "";
  let opened = false;
  const data = await callBridge("listenAndAsk", [lang], (p) => {
    if (p.stage && p.stage.startsWith("Listening")) { setMicState("listening"); haptic("start"); return; }
    // Words appear under the mic while the person speaks ...
    if (p.stage && p.stage.startsWith("“")) { setMicState("listening"); heard = p.stage; setHint(p.stage, true); return; }
    // ... then, once they stop, the answer screen opens straight away
    // (instant answers land in well under a second; others stream in).
    if (p.stage && heard && !opened) {
      opened = true;
      setMicState("idle");
      setHint(t("micHint"));
      openAnswer(heard);
      showLoading();
    }
    if (p.text && opened) showStreaming(p.text);
  });
  setMicState("idle");
  if (data.speech_error !== undefined || (data.error && data.empty_transcription)) {
    if (opened) history.back();
    shakeHint(data.speech_error === 9 ? t("noMic") : data.speech_error !== undefined ? t("noVoice") : t("notHeard"));
    return;
  }
  setHint(t("micHint"));
  if (!opened) openAnswer(heard);
  if (data.transcript) $("answer-title").textContent = "“" + data.transcript + "”";
  // JsBridge times this from the moment the recognizer heard the end of speech.
  renderAnswer(data, { speakNow: true, secs: typeof data.elapsed_s === "number" ? data.elapsed_s : undefined });
}

let meter = null;  // web: { ctx, raf }
function startWebMeter(stream) {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const an = ctx.createAnalyser();
    an.fftSize = 512;
    ctx.createMediaStreamSource(stream).connect(an);
    const buf = new Uint8Array(an.fftSize);
    const tick = () => {
      an.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
      window.onMicLevel(Math.min(1, Math.sqrt(sum / buf.length) * 5).toFixed(2));
      meter.raf = requestAnimationFrame(tick);
    };
    meter = { ctx: ctx, raf: requestAnimationFrame(tick) };
  } catch (e) { meter = null; }
}
function stopWebMeter() {
  if (!meter) return;
  cancelAnimationFrame(meter.raf);
  meter.ctx.close().catch(() => {});
  meter = null;
}

async function webVoice() {
  if (!webVoiceOk()) { shakeHint(t("noMic")); return; }
  if (listening && recorder) { haptic("tap"); recorder.stop(); return; }
  haptic("tap");
  setMicState("starting");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) { setMicState("idle"); shakeHint(t("noMic")); return; }
  stopSpeaking();
  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = async () => {
    startTimer();  // from the end of speaking
    stopWebMeter();
    stream.getTracks().forEach((tr) => tr.stop());
    setMicState("idle");
    setHint(t("micHint"));
    openAnswer("…");
    showLoading();
    const form = new FormData();
    form.append("audio", new Blob(chunks, { type: recorder.mimeType || "audio/webm" }), "question.webm");
    form.append("language", lang);
    form.append("with_audio", "false");
    let data;
    try { data = await (await fetch("/api/voice_ask", { method: "POST", body: form })).json(); }
    catch (e) { data = { error: String(e) }; }
    if (data.transcript) $("answer-title").textContent = "“" + data.transcript + "”";
    renderAnswer(data, { speakNow: true });
  };
  recorder.start();
  startWebMeter(stream);
  setMicState("listening");
  setHint(t("listeningWeb"));
}

// ---------- status ----------
async function setStatus() {
  const dot = $("status-dot");
  if (ON_DEVICE) { dot.className = "dot ok"; $("status-text").textContent = t("offline"); return; }
  try {
    const h = await (await fetch("/api/health")).json();
    const ok = h.ollama && h.model_present;
    dot.className = "dot " + (ok ? "ok" : "bad");
    $("status-text").textContent = ok ? t("ready") : t("noServer");
  } catch (e) {
    dot.className = "dot bad";
    $("status-text").textContent = t("noServer");
  }
}

// ---------- start ----------
$("build-tag").textContent = BUILD_TAG + (ON_DEVICE ? " · on-device" : " · web");
if (!ON_DEVICE && "serviceWorker" in navigator && window.isSecureContext) {
  navigator.serviceWorker.register("static/sw.js").catch(() => {});
}
loadInstantIds();
const start = savedLang();
if (start && LANGS.includes(start)) {
  setLang(start);
  showHome(true);
} else {
  lang = "en";
  $("lang-screen").hidden = false;
}
// Phone: tell the app the first screen is drawn, so it starts loading the AI
// model now (loading it during the first paint delayed the page by seconds).
if (ON_DEVICE && AndroidBridge.pageReady) {
  window.addEventListener("load", () => setTimeout(() => AndroidBridge.pageReady(), 600));
}
