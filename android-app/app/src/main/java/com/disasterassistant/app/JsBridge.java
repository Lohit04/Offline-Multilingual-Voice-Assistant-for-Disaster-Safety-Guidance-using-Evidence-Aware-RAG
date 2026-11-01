package com.disasterassistant.app;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.view.HapticFeedbackConstants;
import android.speech.tts.UtteranceProgressListener;
import android.speech.tts.Voice;
import android.util.Log;
import android.webkit.JavascriptInterface;
import android.webkit.WebView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.regex.Pattern;

/**
 * The standalone, no-server path: everything app.js needs happens on-device
 * through this bridge (retrieval, generation via OnDeviceLlm, TTS via
 * Android's own TextToSpeech) instead of fetch() calls to a remote server.
 * Exposed to the WebView as "AndroidBridge" -- see MainActivity.java.
 *
 * Deliberately scoped down from the server pipeline, not a silent
 * downgrade:
 *  - Retrieval is BM25-only (OnDeviceRetrieval), no dense/embedding
 *    component -- no portable mobile FAISS build, and time didn't allow
 *    building/validating one.
 *  - The live-info structural guard (safety/live_info.py on the server)
 *    is NOT reproduced here -- grounding + abstention are, the extra
 *    safety layer on top of them isn't, given time constraints.
 *  - The LLM is Qwen2.5-0.5B-Instruct (0.5B params) vs. the server's 8B
 *    model -- expect noticeably weaker answers, especially for
 *    translation quality into/out of the 4 non-Hindi Indian languages.
 *
 * Speech-to-text is on-device too (OnDeviceAsr, via whisper-android /
 * whisper.cpp's "small" multilingual model) -- startRecording() and
 * stopRecordingAndAsk() below capture audio with WavRecorder (plain
 * AudioRecord, not the WebView's getUserMedia) and feed the transcript
 * straight into the same handleQuestion() the Text tab uses, so once
 * both models are downloaded once, the Voice tab needs no network either.
 */
public class JsBridge {

    // Per-stage timings: `adb logcat -s DisasterTiming`.
    private static final String TAG = "DisasterTiming";

    static final double BM25_SUFFICIENCY_FLOOR = 3.0;
    private static final int TOP_K = 5;

    // The grounded prompt uses fewer, shorter snippets than TOP_K's full
    // retrieval set -- not a retrieval-quality choice, a hard budget one.
    // The on-device model's KV cache is a fixed 1280 tokens shared between
    // prompt and output (see OnDeviceLlm.MAX_TOKENS); 5 untruncated ~180-
    // word chunks alone can approach or exceed that before the question,
    // instructions, or answer are even counted, which overran the cache
    // and crashed the app rather than failing gracefully. Keeping the
    // evidence well under budget here, plus the MAX_TOKENS backstop in
    // OnDeviceLlm, are two independent layers against the same failure.
    private static final int PROMPT_TOP_K = 3;
    private static final int MAX_EVIDENCE_CHARS_PER_CHUNK = 400;

    private final Context context;
    private final WebView webView;
    // The model and the search index load in the background after the page
    // opens (MainActivity.finishSetup): picture tiles and instant answers
    // never need them, so the app is usable within a couple of seconds.
    // Questions that do need them wait on `enginesReady`.
    private volatile OnDeviceLlm llm;
    private final java.util.concurrent.CountDownLatch enginesReady = new java.util.concurrent.CountDownLatch(1);
    // Loaded on the first voice question, not at startup (see
    // MainActivity.finishSetup). Only touched from `executor`.
    private final String asrModelPath;
    private OnDeviceAsr asr;
    private volatile OnDeviceRetrieval retrieval;
    // Prepared answers matched in milliseconds (null if the asset is missing).
    private final InstantAnswers instant;
    private final TextToSpeech tts;
    private final ExecutorService executor;

    // Set by startRecording(), consumed and cleared by stopRecordingAndAsk().
    // Only ever touched from `executor` (single-threaded), so no
    // synchronization needed -- the WebView can't physically call stop
    // before start's task has already run given the Promise-based
    // request/response pattern app.js uses for both.
    private WavRecorder activeRecorder;

    private static final Pattern DEVANAGARI = Pattern.compile("[\\u0900-\\u097F]");
    private static final Pattern TAMIL = Pattern.compile("[\\u0B80-\\u0BFF]");
    private static final Pattern TELUGU = Pattern.compile("[\\u0C00-\\u0C7F]");
    private static final Pattern KANNADA = Pattern.compile("[\\u0C80-\\u0CFF]");
    private static final Pattern MALAYALAM = Pattern.compile("[\\u0D00-\\u0D7F]");

    public JsBridge(Context context, WebView webView, OnDeviceLlm llm, String asrModelPath,
                     OnDeviceRetrieval retrieval, InstantAnswers instant, TextToSpeech tts, ExecutorService executor) {
        this.instant = instant;
        this.context = context;
        this.webView = webView;
        this.llm = llm;
        this.asrModelPath = asrModelPath;
        this.retrieval = retrieval;
        this.tts = tts;
        this.executor = executor;
        // Load the exact voice speakText() will use (best offline English
        // voice) now, not on the first Listen: measured on the test phone,
        // the first read-aloud took 2.7 s to start when that voice was cold,
        // ~90 ms afterwards.
        // Same for Tamil, Telugu and Hindi: measured cold start 0.5-1.1 s for
        // ta/te. English last, so it is the voice left selected.
        // Runs after the first paint (pageReady -> warmVoices) on its own
        // thread: on the main thread at start-up it held up the page.
        initSpeechCallbacks();
    }

    private void warmVoices() {
        new Thread(() -> {
            long t = SystemClock.elapsedRealtime();
            String[][] warm = {{"ta", "தயார்"}, {"te", "సిద్ధం"},
                    {"hi", "तैयार"}, {"en", "Ready"}};
            for (String[] w : warm) {
                try {
                    Voice v = bestOfflineVoice(w[0]);
                    if (v == null) continue;
                    tts.setVoice(v);
                    tts.synthesizeToFile(w[1], null, new File(context.getCacheDir(), "tts_warmup_" + w[0] + ".wav"), "warmup");
                } catch (Exception ignored) {
                }
            }
            // Leave the chosen language's voice loaded, not English.
            String keep = lastPreparedLanguage;
            if (keep != null) {
                Voice v = bestOfflineVoice(keep);
                if (v != null) tts.setVoice(v);
            }
            mark("voices_warm_ms", SystemClock.elapsedRealtime() - t);
        }, "tts-warmup").start();
    }

    private volatile String lastPreparedLanguage;

    // Start-up timings (ms), readable over USB with startupTimings() -- this
    // phone's ROM hides app log lines.
    static final JSONObject TIMINGS = new JSONObject();

    static synchronized void mark(String key, long ms) {
        try {
            TIMINGS.put(key, ms);
        } catch (Exception ignored) {
        }
    }

    @JavascriptInterface
    public String startupTimings() {
        synchronized (JsBridge.class) {
            return TIMINGS.toString();
        }
    }

    private void initSpeechCallbacks() {
        // Flips the page's "■ Stop" button back to "▶ Listen" when speech ends.
        tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
            @Override public void onStart(String id) {
                if ("answer".equals(id) || "step:0".equals(id) || "step-last:0".equals(id)) {
                    lastTtsStartMs = SystemClock.elapsedRealtime() - speakRequestedAt;
                }
                // speakSteps: tell the page which step is being read so it can light it up.
                if (id != null && id.startsWith("step")) {
                    String n = id.substring(id.indexOf(':') + 1);
                    webView.post(() -> webView.evaluateJavascript(
                            "window.onSpeechStep && window.onSpeechStep(" + n + ")", null));
                }
            }
            @Override public void onDone(String id) {
                if ("answer".equals(id) || (id != null && id.startsWith("step-last:"))) notifySpeechDone();
            }
            @Override public void onError(String id) { notifySpeechDone(); }
            @Override public void onStop(String id, boolean interrupted) { notifySpeechDone(); }
        });
    }

    // --- Voice tab, fully on-device: record -> whisper transcribe -> the
    // same handleQuestion() the Text tab uses. Two bridge calls instead of
    // one because recording start/stop are two separate user taps
    // (app.js's existing mic-button flow), not because the native side
    // needs them split.

    @JavascriptInterface
    public void startRecording(String requestId) {
        executor.execute(() -> {
            JSONObject out = new JSONObject();
            try {
                if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO)
                        != PackageManager.PERMISSION_GRANTED) {
                    throw new SecurityException("Microphone permission was not granted. " +
                            "Allow it in this app's system settings and try again.");
                }
                activeRecorder = new WavRecorder();
                activeRecorder.start(new File(context.getCacheDir(), "on_device_question.wav"));
                out.put("started", true);
            } catch (Exception e) {
                activeRecorder = null;
                try {
                    out.put("started", false);
                    out.put("error", e.getMessage());
                } catch (Exception ignored) {
                }
            }
            postResult(requestId, out);
        });
    }

    @JavascriptInterface
    public void stopRecordingAndAsk(String language, String requestId) {
        executor.execute(() -> {
            JSONObject out = new JSONObject();
            WavRecorder recorder = activeRecorder;
            activeRecorder = null;
            try {
                if (recorder == null) {
                    throw new IllegalStateException("No recording was in progress");
                }
                long t0 = SystemClock.elapsedRealtime();
                File wav = recorder.stop();
                if (asr == null) {
                    postStage(requestId, "Loading speech recognition (first time only)…");
                    long tl = SystemClock.elapsedRealtime();
                    asr = new OnDeviceAsr(context, asrModelPath);
                    Log.w(TAG, "asr_load_ms=" + (SystemClock.elapsedRealtime() - tl));
                }
                postStage(requestId, "Understanding your speech…");
                String transcript;
                long ta = SystemClock.elapsedRealtime();
                try {
                    // Told the language (from the page's chip), 4 threads. Measured on
                    // the test phone (Snapdragon 768G: 2 fast + 6 slow cores),
                    // Tamil clip, tiny model: 4.2 s with the language given and 4
                    // threads; 7.6 s on auto-detect; 27.8 s with 8 threads (the slow
                    // cores hold the fast ones back).
                    String lang = "auto".equals(language) ? "auto" : language;
                    transcript = asr.transcribe(wav.getAbsolutePath(), lang, 4);
                } finally {
                    Log.w(TAG, "asr_transcribe_ms=" + (SystemClock.elapsedRealtime() - ta)
                            + " wav_bytes=" + wav.length());
                    wav.delete();
                }
                if (transcript == null || transcript.trim().isEmpty()) {
                    out.put("error", "Could not hear a clear question. Please try again, " +
                            "speaking close to the microphone.");
                    out.put("empty_transcription", true);
                } else {
                    String heard = transcript.trim();
                    out = handleQuestion(heard, language, requestId);
                    out.put("transcript", heard);
                }
                Log.w(TAG, "voice_total_ms=" + (SystemClock.elapsedRealtime() - t0));
                putTiming(out, t0);
            } catch (Exception e) {
                try {
                    out.put("error", "On-device speech recognition failed: " + e.getMessage());
                } catch (Exception ignored) {
                }
            }
            postResult(requestId, out);
        });
    }

    // --- Voice, fast path: Android's own on-device speech recognizer ---
    //
    // Why (2026-10-26): with instant answers the voice round trip on the
    // phone was still >10 s, nearly all of it Whisper transcribing the whole
    // recording on the CPU after the user tapped stop. Android's recognizer
    // (Google's on-device engine; this phone has Android System Intelligence)
    // transcribes WHILE the user speaks and detects the end of speech by
    // itself, so the text is ready about a second after they stop talking,
    // with no second tap. Whisper (startRecording/stopRecordingAndAsk above)
    // stays as the fallback when this engine is missing or lacks the
    // language offline -- the page switches automatically on an error.

    private final Handler main = new Handler(Looper.getMainLooper());
    private SpeechRecognizer recognizer;

    @JavascriptInterface
    public boolean nativeSpeechAvailable() {
        return SpeechRecognizer.isRecognitionAvailable(context);
    }

    // Locales to try, in order, on the on-device engine. Measured on the test
    // phone (checkRecognitionSupport): only en-US is installed by default;
    // en-IN and hi-IN are downloadable (requestSpeechPacks); Tamil, Telugu,
    // Kannada and Malayalam are not offered by the engine at all -- those
    // languages use Whisper (the page goes there directly).
    private static String[] speechLocales(String language) {
        switch (language) {
            case "en": return new String[]{"en-IN", "en-US"};
            case "hi": return new String[]{"hi-IN"};
            // ta/te/kn/ml: no engine apps can use offers them offline on the
            // test phone (2026-10-26). Both engines' checkRecognitionSupport
            // list en/hi only; a Tamil pack downloaded in the Google app is
            // private to that app. The page asks these users to type; see
            // docs/implementation_status.md Session 13 for the Whisper results.
            default: return new String[]{};
        }
    }

    @JavascriptInterface
    public boolean nativeSpeechSupports(String language) {
        return speechLocales(language).length > 0 && nativeSpeechAvailable();
    }

    @JavascriptInterface
    public void listenAndAsk(String language, String requestId) {
        main.post(() -> startNative(speechLocales(language), 0, requestId));
    }

    // Runs on the main thread. On "language unavailable/unsupported" (codes
    // 12/13, returned before the user speaks) it moves straight on to the
    // next locale, so English falls from en-IN to the installed en-US
    // without the user noticing.
    private void startNative(String[] locales, int idx, String requestId) {
        if (idx >= locales.length) {
            JSONObject out = new JSONObject();
            put(out, "error", "No offline speech engine for this language on this phone.");
            put(out, "speech_error", SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE);
            postResult(requestId, out);
            return;
        }
        try {
            if (recognizer != null) recognizer.destroy();
            boolean useDefault = locales[idx].startsWith("default:");
            String tag = useDefault ? locales[idx].substring("default:".length()) : locales[idx];
            boolean onDevice = !useDefault && Build.VERSION.SDK_INT >= 31
                    && SpeechRecognizer.isOnDeviceRecognitionAvailable(context);
            recognizer = onDevice ? SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
                    : SpeechRecognizer.createSpeechRecognizer(context);
            Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, tag);
            intent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true);
            intent.putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true);
            final long[] endOfSpeech = {0};
            final String engine = (onDevice ? "on-device " : "offline-preferred ") + tag;
            recognizer.setRecognitionListener(new RecognitionListener() {
                @Override public void onReadyForSpeech(Bundle b) { postStage(requestId, "Listening… speak now"); }
                @Override public void onBeginningOfSpeech() { }
                @Override public void onRmsChanged(float v) { postMicLevel(v); }
                @Override public void onBufferReceived(byte[] b) { }
                @Override public void onEndOfSpeech() {
                    endOfSpeech[0] = SystemClock.elapsedRealtime();
                    postStage(requestId, "Finding the answer…");
                }
                @Override public void onPartialResults(Bundle b) {
                    String heard = first(b);
                    if (heard != null && !heard.isEmpty()) postStage(requestId, "“" + heard + "”");
                }
                @Override public void onEvent(int t, Bundle b) { }
                @Override public void onResults(Bundle b) {
                    String heard = first(b);
                    long spokeEnd = endOfSpeech[0] == 0 ? SystemClock.elapsedRealtime() : endOfSpeech[0];
                    executor.execute(() -> {
                        JSONObject out;
                        if (heard == null || heard.trim().isEmpty()) {
                            out = new JSONObject();
                            put(out, "error", "Could not hear a clear question. Please try again.");
                            put(out, "empty_transcription", true);
                        } else {
                            out = handleQuestion(heard.trim(), "auto", requestId);
                            put(out, "transcript", heard.trim());
                        }
                        // What the user actually waits: from the moment they stop talking.
                        putTiming(out, spokeEnd);
                        put(out, "speech_engine", engine);
                        postResult(requestId, out);
                    });
                }
                @Override public void onError(int code) {
                    // Network errors too: an engine that needs internet must
                    // not count as offline speech -- move on (and finally to
                    // Whisper on the page).
                    if (code == SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE
                            || code == SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED
                            || code == SpeechRecognizer.ERROR_NETWORK
                            || code == SpeechRecognizer.ERROR_NETWORK_TIMEOUT
                            || code == SpeechRecognizer.ERROR_SERVER) {
                        main.post(() -> startNative(locales, idx + 1, requestId));
                        return;
                    }
                    JSONObject out = new JSONObject();
                    if (code == SpeechRecognizer.ERROR_NO_MATCH || code == SpeechRecognizer.ERROR_SPEECH_TIMEOUT) {
                        put(out, "error", "I didn't catch that. Tap the microphone and try again.");
                        put(out, "empty_transcription", true);
                    } else {
                        put(out, "error", "Speech engine error " + code);
                        put(out, "speech_error", code);
                    }
                    postResult(requestId, out);
                }
            });
            recognizer.startListening(intent);
        } catch (Exception e) {
            JSONObject out = new JSONObject();
            put(out, "error", "Speech engine unavailable: " + e.getMessage());
            put(out, "speech_error", -1);
            postResult(requestId, out);
        }
    }

    /**
     * Asks Android to download the on-device English (India) and Hindi
     * speech packs (Android 13+), only for packs that are neither installed
     * nor already downloading, and at most once per pack per install: the
     * system shows a "Download ... (113 MB)" dialog for each request, and
     * popping that over the app on every launch is not acceptable.
     */
    public void requestSpeechPacks() {
        if (Build.VERSION.SDK_INT < 33) return;
        android.content.SharedPreferences prefs = context.getSharedPreferences("speech", Context.MODE_PRIVATE);
        main.post(() -> {
            try {
                if (!SpeechRecognizer.isOnDeviceRecognitionAvailable(context)) return;
                SpeechRecognizer rec = SpeechRecognizer.createOnDeviceSpeechRecognizer(context);
                Intent probe = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
                probe.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
                rec.checkRecognitionSupport(probe, executor, new android.speech.RecognitionSupportCallback() {
                    @Override public void onSupportResult(android.speech.RecognitionSupport r) {
                        main.post(() -> {
                            for (String tag : new String[]{"en-IN", "hi-IN"}) {
                                boolean have = r.getInstalledOnDeviceLanguages().contains(tag)
                                        || r.getPendingOnDeviceLanguages().contains(tag);
                                if (have || prefs.getBoolean("asked_" + tag, false)) continue;
                                if (!r.getSupportedOnDeviceLanguages().contains(tag)) continue;
                                prefs.edit().putBoolean("asked_" + tag, true).apply();
                                Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
                                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
                                intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, tag);
                                rec.triggerModelDownload(intent);
                            }
                            main.postDelayed(rec::destroy, 5000);
                        });
                    }
                    @Override public void onError(int error) {
                        main.post(rec::destroy);
                    }
                });
            } catch (Exception e) {
                Log.w(TAG, "speech pack download request failed: " + e.getMessage());
            }
        });
    }

    /**
     * What each speech engine on this phone can do offline (Android 13+),
     * as JSON, for diagnosing "language unavailable" errors. Reported per
     * engine: languages installed on-device, downloadable, and online-only.
     */
    @JavascriptInterface
    public void speechSupport(String requestId) {
        main.post(() -> {
            JSONObject out = new JSONObject();
            if (Build.VERSION.SDK_INT < 33) {
                put(out, "error", "needs Android 13");
                postResult(requestId, out);
                return;
            }
            Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
            intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
            final int[] pending = {2};
            String[] names = {"on_device", "default"};
            SpeechRecognizer[] recs = {
                    SpeechRecognizer.isOnDeviceRecognitionAvailable(context)
                            ? SpeechRecognizer.createOnDeviceSpeechRecognizer(context) : null,
                    SpeechRecognizer.createSpeechRecognizer(context)};
            for (int i = 0; i < 2; i++) {
                final String name = names[i];
                final SpeechRecognizer rec = recs[i];
                if (rec == null) {
                    put(out, name, "not available");
                    if (--pending[0] == 0) postResult(requestId, out);
                    continue;
                }
                rec.checkRecognitionSupport(intent, java.util.concurrent.Executors.newSingleThreadExecutor(),
                        new android.speech.RecognitionSupportCallback() {
                            @Override public void onSupportResult(android.speech.RecognitionSupport r) {
                                JSONObject o = new JSONObject();
                                put(o, "installed", new JSONArray(r.getInstalledOnDeviceLanguages()));
                                put(o, "pending", new JSONArray(r.getPendingOnDeviceLanguages()));
                                put(o, "downloadable", new JSONArray(r.getSupportedOnDeviceLanguages()));
                                put(o, "online", new JSONArray(r.getOnlineLanguages()));
                                finish(o);
                            }
                            @Override public void onError(int error) {
                                JSONObject o = new JSONObject();
                                put(o, "error", error);
                                finish(o);
                            }
                            private void finish(JSONObject o) {
                                main.post(() -> {
                                    put(out, name, o);
                                    rec.destroy();
                                    if (--pending[0] == 0) postResult(requestId, out);
                                });
                            }
                        });
            }
        });
    }

    /** Stops listening early; the result still arrives through onResults. */
    @JavascriptInterface
    public void stopListening() {
        main.post(() -> {
            if (recognizer != null) recognizer.stopListening();
        });
    }

    private static String first(Bundle b) {
        if (b == null) return null;
        java.util.ArrayList<String> r = b.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
        return r == null || r.isEmpty() ? null : r.get(0);
    }

    private static void put(JSONObject o, String k, Object v) {
        try {
            o.put(k, v);
        } catch (Exception ignored) {
        }
    }

    @JavascriptInterface
    public void askQuestion(String question, String language, String requestId) {
        executor.execute(() -> {
            long t0 = SystemClock.elapsedRealtime();
            JSONObject result = handleQuestion(question, language, requestId);
            Log.w(TAG, "text_total_ms=" + (SystemClock.elapsedRealtime() - t0));
            putTiming(result, t0);
            postResult(requestId, result);
        });
    }

    // Shown under the answer ("Answered in 9.8 s on this phone · GPU") --
    // this phone's ROM hides app log lines, so the page is where timings
    // are visible, and it is honest evidence for a demo.
    private void putTiming(JSONObject out, long startMs) {
        try {
            out.put("elapsed_s", Math.round((SystemClock.elapsedRealtime() - startMs) / 100.0) / 10.0);
            if (llm != null) out.put("backend", llm.backend);
        } catch (Exception ignored) {
        }
    }

    // Run once, when the page reports it has drawn its first screen (or by
    // MainActivity's fallback timer): starts loading the model, whose GPU
    // set-up otherwise competes with the page's first paint.
    private final java.util.concurrent.atomic.AtomicReference<Runnable> onPageReady = new java.util.concurrent.atomic.AtomicReference<>();

    public void setOnPageReady(Runnable r) {
        onPageReady.set(r);
    }

    @JavascriptInterface
    public void pageReady() {
        Runnable r = onPageReady.getAndSet(null);
        if (r == null) return;
        mark("page_ready_ms", SystemClock.elapsedRealtime() - MainActivity.createdAt);
        r.run();
        warmVoices();
    }

    /** Called from the background loader once the model and search index are in memory. */
    public void setEngines(OnDeviceLlm llm, OnDeviceRetrieval retrieval) {
        mark("engines_ready_ms", SystemClock.elapsedRealtime() - MainActivity.createdAt);
        this.llm = llm;
        this.retrieval = retrieval;
        enginesReady.countDown();
    }

    /** Loading failed: release waiting questions so they report an error instead of hanging. */
    public void enginesFailed() {
        enginesReady.countDown();
    }

    /** Loads speech recognition on the worker thread, ahead of the first voice question. */
    public void preloadAsr() {
        executor.execute(() -> {
            if (asr != null) return;
            try {
                long t = SystemClock.elapsedRealtime();
                asr = new OnDeviceAsr(context, asrModelPath);
                // One throwaway transcription of a real 2 s spoken Tamil clip
                // (assets/data/warmup_ta.wav, the project's own TTS voice): on the
                // test phone the first real transcription took 34-94 s cold vs
                // ~5 s warm, and warming with silence did not help (whisper.cpp
                // skips most of the work on silence).
                File clip = new File(context.getCacheDir(), "warmup_ta.wav");
                try (java.io.InputStream in = context.getAssets().open("data/warmup_ta.wav");
                     java.io.FileOutputStream out = new java.io.FileOutputStream(clip)) {
                    byte[] buf = new byte[65536];
                    int r;
                    while ((r = in.read(buf)) != -1) out.write(buf, 0, r);
                }
                asr.transcribe(clip.getAbsolutePath(), "ta", 4);
                Log.w(TAG, "asr_preload_ms=" + (SystemClock.elapsedRealtime() - t));
            } catch (Exception e) {
                // Not fatal: stopRecordingAndAsk retries the load and reports it.
                Log.w(TAG, "asr preload failed: " + e.getMessage());
            }
        });
    }

    /** Times on-phone Whisper on a WAV already in the app's cache dir -- for latency tests. */
    @JavascriptInterface
    public void debugTranscribe(String fileName, String requestId) {
        executor.execute(() -> {
            JSONObject out = new JSONObject();
            try {
                if (asr == null) asr = new OnDeviceAsr(context, asrModelPath);
                long t = SystemClock.elapsedRealtime();
                String text = asr.transcribe(new File(context.getCacheDir(), fileName).getAbsolutePath(), "ta", 4);
                put(out, "ms", SystemClock.elapsedRealtime() - t);
                put(out, "text", text);
            } catch (Exception e) {
                put(out, "error", e.getMessage());
            }
            postResult(requestId, out);
        });
    }

    private final java.util.Map<String, OnDeviceAsr> debugModels = new java.util.HashMap<>();

    /** Latency experiments: whisper model file (in files/), language, threads. */
    @JavascriptInterface
    public void debugTranscribe2(String fileName, String modelFile, String language, int threads, String requestId) {
        executor.execute(() -> {
            JSONObject out = new JSONObject();
            try {
                OnDeviceAsr m = debugModels.get(modelFile);
                if (m == null) {
                    long tl = SystemClock.elapsedRealtime();
                    m = new OnDeviceAsr(context, new File(context.getFilesDir(), modelFile).getAbsolutePath());
                    debugModels.put(modelFile, m);
                    put(out, "load_ms", SystemClock.elapsedRealtime() - tl);
                }
                long t = SystemClock.elapsedRealtime();
                String text = m.transcribe(new File(context.getCacheDir(), fileName).getAbsolutePath(), language, threads);
                put(out, "ms", SystemClock.elapsedRealtime() - t);
                put(out, "text", text);
            } catch (Exception e) {
                put(out, "error", String.valueOf(e));
            }
            postResult(requestId, out);
        });
    }

    /** Installed text-to-speech voices for a language, for diagnosing read-aloud. */
    @JavascriptInterface
    public String ttsVoices(String language) {
        JSONArray arr = new JSONArray();
        try {
            for (Voice v : tts.getVoices()) {
                if (!v.getLocale().getLanguage().equals(localeFor(language).getLanguage())) continue;
                JSONObject o = new JSONObject();
                o.put("name", v.getName());
                o.put("quality", v.getQuality());
                o.put("network", v.isNetworkConnectionRequired());
                o.put("not_installed", v.getFeatures() != null
                        && v.getFeatures().contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED));
                arr.put(o);
            }
        } catch (Exception e) {
            return "error: " + e;
        }
        return arr.toString();
    }

    /** Prepared situation ids, for the home screen's hazard tiles. */
    @JavascriptInterface
    public String instantIds() {
        return instant == null ? "[]" : instant.ids().toString();
    }

    /** A prepared situation's answer by id (the hazard tiles): no matching, no model. */
    @JavascriptInterface
    public String instantById(String id, String language) {
        JSONObject out = new JSONObject();
        JSONObject it = instant == null ? null : instant.byId(id);
        if (it == null) {
            put(out, "error", "Unknown situation.");
            return out.toString();
        }
        JSONObject answers = it.optJSONObject("answers");
        boolean have = answers != null && answers.has(language);
        put(out, "text", answers == null ? "" : answers.optString(have ? language : "en"));
        put(out, "answer_language", have ? language : "en");
        put(out, "abstained", false);
        put(out, "cached", true);
        put(out, "situation", id);
        put(out, "sources", it.optJSONArray("sources"));
        return out.toString();
    }

    /** Which prepared situation (if any) a question matches, without answering it -- for parity tests. */
    @JavascriptInterface
    public String instantMatchId(String question) {
        InstantAnswers.Match m = instant == null ? null : instant.match(question);
        return m == null ? "" : m.intent.optString("id");
    }

    // 16 kHz mono 16-bit PCM WAV of `samples` zero samples (44-byte header).
    private static void writeSilentWav(File f, int samples) throws java.io.IOException {
        int dataLen = samples * 2;
        java.nio.ByteBuffer b = java.nio.ByteBuffer.allocate(44 + dataLen).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        b.put("RIFF".getBytes()).putInt(36 + dataLen).put("WAVE".getBytes()).put("fmt ".getBytes())
                .putInt(16).putShort((short) 1).putShort((short) 1).putInt(16000).putInt(32000)
                .putShort((short) 2).putShort((short) 16).put("data".getBytes()).putInt(dataLen);
        try (java.io.FileOutputStream out = new java.io.FileOutputStream(f)) {
            out.write(b.array());
        }
    }

    /** Releases the speech model if it was ever loaded. Not exposed to JS. */
    public void close() {
        if (asr != null) asr.close();
    }

    // JSONObject.quote() yields a valid JS string literal (it escapes
    // quotes, backslashes, all control characters and U+2028/2029), unlike
    // the earlier hand-rolled replace() chain, which missed \r and others.
    private void postResult(String requestId, JSONObject result) {
        String js = "window.onBridgeResult(" + JSONObject.quote(requestId) + ", "
                + JSONObject.quote(result.toString()) + ")";
        webView.post(() -> webView.evaluateJavascript(js, null));
    }

    // Loudness while listening (the recognizer's RMS in dB, roughly -2..10),
    // so the mic ring on the page visibly moves with the user's voice.
    private long lastLevelPost = 0;

    private void postMicLevel(float rmsDb) {
        long now = SystemClock.elapsedRealtime();
        if (now - lastLevelPost < 50) return;
        lastLevelPost = now;
        float level = Math.max(0f, Math.min(1f, (rmsDb + 2f) / 12f));
        String js = "window.onMicLevel && window.onMicLevel(" + String.format(Locale.US, "%.2f", level) + ")";
        webView.post(() -> webView.evaluateJavascript(js, null));
    }

    /** Replaces the waiting label with what is actually happening now. */
    private void postStage(String requestId, String label) {
        String js = "window.onBridgeStage && window.onBridgeStage(" + JSONObject.quote(requestId) + ", "
                + JSONObject.quote(label) + ")";
        webView.post(() -> webView.evaluateJavascript(js, null));
    }

    // Partial answer text while the model is still writing. Throttled:
    // the model emits a piece every few tens of ms, and re-rendering the
    // page that often only slows the phone down further.
    private long lastPartialPost = 0;

    private void postPartial(String requestId, String textSoFar) {
        long now = SystemClock.elapsedRealtime();
        if (now - lastPartialPost < 150) return;
        lastPartialPost = now;
        String js = "window.onBridgePartial && window.onBridgePartial(" + JSONObject.quote(requestId) + ", "
                + JSONObject.quote(textSoFar) + ")";
        webView.post(() -> webView.evaluateJavascript(js, null));
    }

    private JSONObject handleQuestion(String question, String requestedLanguage, String requestId) {
        JSONObject out = new JSONObject();
        try {
            String language = "auto".equals(requestedLanguage) ? detectLanguageFromScript(question) : requestedLanguage;

            // Instant path: a question matching a prepared situation is
            // answered by lookup (milliseconds), with that situation's
            // stored, cited answer -- no translation or generation step.
            InstantAnswers.Match im = instant == null ? null : instant.match(question);
            if (im != null) {
                JSONObject answers = im.intent.getJSONObject("answers");
                boolean have = answers.has(language);
                out.put("text", have ? answers.getString(language) : answers.getString("en"));
                out.put("answer_language", have ? language : "en");
                out.put("abstained", false);
                out.put("abstention_reason", JSONObject.NULL);
                out.put("cached", true);
                out.put("sources", im.intent.getJSONArray("sources"));
                Log.w(TAG, "instant_match score=" + im.score + " intent=" + im.intent.optString("id"));
                return out;
            }

            if (enginesReady.getCount() > 0) {
                postStage(requestId, "Getting ready… (first question after opening)");
                enginesReady.await();
            }
            if (llm == null || retrieval == null) throw new IllegalStateException("on-device model failed to load");

            String searchText = question;
            if (!"en".equals(language)) {
                postStage(requestId, "Understanding your question…");
                long tt = SystemClock.elapsedRealtime();
                searchText = llm.generate(translatePrompt(question, language)).trim();
                Log.w(TAG, "translate_question_ms=" + (SystemClock.elapsedRealtime() - tt) + " lang=" + language);
                out.put("search_question", searchText);
            }

            postStage(requestId, "Searching official guidance…");
            long tr = SystemClock.elapsedRealtime();
            List<OnDeviceRetrieval.Result> results = retrieval.search(searchText, TOP_K);
            boolean sufficient = !results.isEmpty() && results.get(0).score >= BM25_SUFFICIENCY_FLOOR;
            Log.w(TAG, "retrieval_ms=" + (SystemClock.elapsedRealtime() - tr) + " sufficient=" + sufficient);

            out.put("answer_language", language);
            out.put("abstained", !sufficient);

            if (!sufficient) {
                out.put("text", abstentionMessage(language));
                out.put("abstention_reason", "bm25_score_below_floor");
                out.put("sources", new JSONArray());
            } else {
                // Evidence for the model comes from one hazard only -- the top
                // hit's. Passages about other hazards in the same prompt are
                // what produced fire advice in a flood answer.
                String topCategory = results.get(0).chunk.disasterCategory;
                List<OnDeviceRetrieval.Result> promptEvidence = new java.util.ArrayList<>();
                for (OnDeviceRetrieval.Result r : results) {
                    if (promptEvidence.size() < PROMPT_TOP_K && topCategory.equals(r.chunk.disasterCategory)) {
                        promptEvidence.add(r);
                    }
                }
                String prompt = groundedPrompt(searchText, promptEvidence, language);
                postStage(requestId, "Writing your answer…");
                long tg = SystemClock.elapsedRealtime();
                String answer = llm.generateStreaming(prompt, soFar -> postPartial(requestId, soFar)).trim();
                Log.w(TAG, "generate_ms=" + (SystemClock.elapsedRealtime() - tg)
                        + " prompt_tokens=" + llm.sizeInTokens(prompt) + " backend=" + llm.backend
                        + " answer_chars=" + answer.length());

                // The 1B model often ignores "write your answer in Tamil"
                // and answers in English. Checked by script, not trusted:
                // if the answer isn't in the asked-for language, translate it.
                if (!"en".equals(language) && !language.equals(detectLanguageFromScript(answer))) {
                    postStage(requestId, "Translating the answer to " + languageName(language) + "…");
                    long tx = SystemClock.elapsedRealtime();
                    answer = llm.generateStreaming(translateAnswerPrompt(answer, language),
                            soFar -> postPartial(requestId, soFar)).trim();
                    Log.w(TAG, "translate_answer_ms=" + (SystemClock.elapsedRealtime() - tx));
                    out.put("machine_translated", true);
                }
                out.put("text", answer);
                out.put("abstention_reason", JSONObject.NULL);

                JSONArray sources = new JSONArray();
                for (OnDeviceRetrieval.Result r : results) {
                    JSONObject s = new JSONObject();
                    s.put("document_id", r.chunk.documentId);
                    s.put("page", r.chunk.page);
                    s.put("disaster_category", r.chunk.disasterCategory);
                    s.put("dense_score", 0.0);
                    s.put("bm25_score", r.score);
                    s.put("text", r.chunk.text);
                    sources.put(s);
                }
                out.put("sources", sources);
                // Not spoken automatically any more: the page has a Listen
                // button (and auto-presses it for voice questions).
            }
        } catch (Exception e) {
            try {
                out.put("error", "On-device processing failed: " + e.getMessage());
            } catch (Exception ignored) {
            }
        }
        return out;
    }

    // --- Read-aloud (the page's Listen button) ---

    /**
     * Speaks the answer with Android's own text-to-speech. Best-effort and
     * separate from answering: a missing voice can never cost the user the
     * written answer (an earlier version spoke inside the answer path, and
     * a TTS failure threw away a good answer).
     *
     * @return false if this language has no installed voice; the page then
     *         says so on the button instead of failing silently.
     */
    @JavascriptInterface
    public boolean speakText(String text, String language) {
        try {
            Voice voice = bestOfflineVoice(language);
            if (voice != null) {
                tts.setVoice(voice);
            } else {
                int r = tts.setLanguage(localeFor(language));
                if (r == TextToSpeech.LANG_MISSING_DATA || r == TextToSpeech.LANG_NOT_SUPPORTED) return false;
            }
            // Slightly slower than default: instructions heard under stress.
            // Normal speed: answers need to be heard quickly (was 0.92).
            tts.setSpeechRate(1.0f);
            speakRequestedAt = SystemClock.elapsedRealtime();
            return tts.speak(forSpeech(text), TextToSpeech.QUEUE_FLUSH, null, "answer") == TextToSpeech.SUCCESS;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Speaks the answer one step at a time (queued, so there is no gap the
     * user can hear) and reports each step's start to the page through
     * window.onSpeechStep(i), which highlights the step being read.
     *
     * @param stepsJson JSON array of the step texts, in screen order.
     */
    @JavascriptInterface
    public boolean speakSteps(String stepsJson, String language) {
        try {
            JSONArray steps = new JSONArray(stepsJson);
            if (steps.length() == 0) return false;
            Voice voice = bestOfflineVoice(language);
            if (voice != null) {
                tts.setVoice(voice);
            } else {
                int r = tts.setLanguage(localeFor(language));
                if (r == TextToSpeech.LANG_MISSING_DATA || r == TextToSpeech.LANG_NOT_SUPPORTED) return false;
            }
            tts.setSpeechRate(1.0f);
            speakRequestedAt = SystemClock.elapsedRealtime();
            int last = steps.length() - 1;
            for (int i = 0; i <= last; i++) {
                String id = (i == last ? "step-last:" : "step:") + i;
                int mode = i == 0 ? TextToSpeech.QUEUE_FLUSH : TextToSpeech.QUEUE_ADD;
                if (tts.speak(forSpeech(steps.getString(i)), mode, null, id) != TextToSpeech.SUCCESS) return i > 0;
            }
            return true;
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Native haptic tick for the page ("tap", "start", "success", "error").
     * Uses the view's haptic feedback, which needs no permission and follows
     * the user's own touch-vibration setting.
     */
    @JavascriptInterface
    public void haptic(String kind) {
        final int c;
        switch (kind) {
            case "success": c = Build.VERSION.SDK_INT >= 30 ? HapticFeedbackConstants.CONFIRM : HapticFeedbackConstants.VIRTUAL_KEY; break;
            case "error": c = Build.VERSION.SDK_INT >= 30 ? HapticFeedbackConstants.REJECT : HapticFeedbackConstants.LONG_PRESS; break;
            case "start": c = HapticFeedbackConstants.LONG_PRESS; break;
            default: c = HapticFeedbackConstants.VIRTUAL_KEY;
        }
        webView.post(() -> webView.performHapticFeedback(c));
    }

    private volatile long speakRequestedAt;
    private volatile long lastTtsStartMs = -1;

    /** Milliseconds between the Listen request and audio starting, last time (for timing tests). */
    @JavascriptInterface
    public long lastTtsStartMs() {
        return lastTtsStartMs;
    }

    /**
     * Loads a language's voice ahead of the first answer (called when the
     * user taps a language chip). The speech engine keeps one voice loaded,
     * so the first answer after a switch waited ~0.8-1.3 s on the test phone
     * and later ones ~0.01 s.
     */
    @JavascriptInterface
    public void prepareVoice(String language) {
        lastPreparedLanguage = language;
        main.post(() -> {
            try {
                Voice v = bestOfflineVoice(language);
                if (v == null) return;
                tts.setVoice(v);
                tts.synthesizeToFile(".", null, new File(context.getCacheDir(), "tts_prepare.wav"), "warmup");
            } catch (Exception ignored) {
            }
        });
    }

    @JavascriptInterface
    public void stopSpeaking() {
        try {
            tts.stop();
        } catch (Exception ignored) {
        }
        notifySpeechDone();
    }

    private void notifySpeechDone() {
        webView.post(() -> webView.evaluateJavascript("window.onSpeechDone && window.onSpeechDone()", null));
    }

    // The installed voice that sounds best for this language AND needs no
    // network: Indian-English/Indian-language voices first, then highest
    // quality. Network voices are skipped -- the app must work offline.
    private Voice bestOfflineVoice(String language) {
        Locale want = localeFor(language);
        Voice best = null;
        int bestScore = -1;
        java.util.Set<Voice> voices = tts.getVoices();
        if (voices == null) return null;
        for (Voice v : voices) {
            if (v.isNetworkConnectionRequired()) continue;
            if (v.getFeatures() != null && v.getFeatures().contains(TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED)) continue;
            Locale l = v.getLocale();
            if (!l.getLanguage().equals(want.getLanguage())) continue;
            int score = v.getQuality() + ("IN".equals(l.getCountry()) ? 1000 : 0);
            if (score > bestScore) {
                bestScore = score;
                best = v;
            }
        }
        return best;
    }

    // Markdown the model sometimes emits ("**Stay calm**") would be read
    // out as "asterisk asterisk".
    private static String forSpeech(String text) {
        return text.replaceAll("[*_#`]", "").replaceAll("\\s+\\n", "\n");
    }

    private static Locale localeFor(String language) {
        switch (language) {
            case "hi": return new Locale("hi", "IN");
            case "ta": return new Locale("ta", "IN");
            case "te": return new Locale("te", "IN");
            case "kn": return new Locale("kn", "IN");
            case "ml": return new Locale("ml", "IN");
            default: return new Locale("en", "IN");
        }
    }

    static String detectLanguageFromScript(String text) {
        int hi = count(DEVANAGARI, text), ta = count(TAMIL, text), te = count(TELUGU, text),
                kn = count(KANNADA, text), ml = count(MALAYALAM, text);
        int max = Math.max(hi, Math.max(ta, Math.max(te, Math.max(kn, ml))));
        if (max == 0) return "en";
        if (max == hi) return "hi";
        if (max == ta) return "ta";
        if (max == te) return "te";
        if (max == kn) return "kn";
        return "ml";
    }

    private static int count(Pattern p, String text) {
        java.util.regex.Matcher m = p.matcher(text);
        int n = 0;
        while (m.find()) n++;
        return n;
    }

    private static String languageName(String code) {
        switch (code) {
            case "hi": return "Hindi";
            case "ta": return "Tamil";
            case "te": return "Telugu";
            case "kn": return "Kannada";
            case "ml": return "Malayalam";
            default: return "English";
        }
    }

    private static String translatePrompt(String text, String sourceLanguage) {
        return "Translate the following " + languageName(sourceLanguage) + " text into English. " +
                "Output ONLY the English translation, nothing else.\n\n" +
                languageName(sourceLanguage) + " text: " + text + "\n\nEnglish translation:";
    }

    private static String translateAnswerPrompt(String englishAnswer, String targetLanguage) {
        String lang = languageName(targetLanguage);
        return "Translate the following English safety instructions into " + lang + ". Keep the numbering. " +
                "Output ONLY the " + lang + " translation, nothing else.\n\n" +
                "English: " + englishAnswer + "\n\n" + lang + " translation:";
    }

    private static String groundedPrompt(String question, List<OnDeviceRetrieval.Result> results, String answerLanguage) {
        StringBuilder evidence = new StringBuilder();
        int i = 1;
        for (OnDeviceRetrieval.Result r : results) {
            evidence.append("[").append(i++).append("] (source: ").append(r.chunk.documentId)
                    .append(", page ").append(r.chunk.page).append(")\n")
                    .append(truncate(r.chunk.text, MAX_EVIDENCE_CHARS_PER_CHUNK)).append("\n\n");
        }
        String lang = languageName(answerLanguage);
        return "You are a disaster-safety information assistant. The person asking may be in the middle of " +
                "an actual emergency, frightened, and reading or listening under stress. Answer using ONLY the " +
                "evidence below, which is in English.\n\n" +
                "Rules: base your answer strictly on the evidence; never invent current shelter/road/weather " +
                "facts; use simple plain words and short sentences; number steps if there is more than one " +
                "action, at most 5 short steps; do not pad a thin answer with vague advice; " +
                "write your answer in " + lang + ".\n\n" +
                "Evidence:\n" + evidence +
                "Question: " + question + "\n\nAnswer (in " + lang + "):";
    }

    private static String truncate(String text, int maxChars) {
        if (text.length() <= maxChars) return text;
        // Cut at the last space before the limit rather than mid-word --
        // cosmetic, but a half-written word at a hard cutoff reads like a
        // bug the first time someone notices it.
        int cut = text.lastIndexOf(' ', maxChars);
        if (cut <= 0) cut = maxChars;
        return text.substring(0, cut) + "...";
    }

    private static String abstentionMessage(String language) {
        switch (language) {
            case "hi":
                return "मेरे पास इसका उत्तर देने के लिए विश्वसनीय जानकारी नहीं है। कृपया अधिकारियों से संपर्क करें। आपातकाल में 112 डायल करें।";
            case "ta":
                return "இதற்கு பதிலளிக்க நம்பகமான தகவல் இல்லை. தயவுசெய்து அதிகாரிகளை தொடர்பு கொள்ளவும். அவசரநிலையில் 112-ஐ அழைக்கவும்.";
            case "te":
                return "దీనికి సమాధానం ఇవ్వడానికి నాకు విశ్వసనీయ సమాచారం లేదు. దయచేసి అధికారులను సంప్రదించండి. అత్యవసర పరిస్థితిలో 112కి కాల్ చేయండి.";
            case "kn":
                return "ಇದಕ್ಕೆ ಉತ್ತರಿಸಲು ನನಗೆ ವಿಶ್ವಾಸಾರ್ಹ ಮಾಹಿತಿ ಇಲ್ಲ. ದಯವಿಟ್ಟು ಅಧಿಕಾರಿಗಳನ್ನು ಸಂಪರ್ಕಿಸಿ. ತುರ್ತು ಸಂದರ್ಭದಲ್ಲಿ 112 ಗೆ ಕರೆ ಮಾಡಿ.";
            case "ml":
                return "ഇതിന് ഉത്തരം നൽകാൻ എനിക്ക് വിശ്വസനീയമായ വിവരങ്ങൾ ഇല്ല. ദയവായി അധികാരികളെ ബന്ധപ്പെടുക. അടിയന്തര ഘട്ടത്തിൽ 112 എന്ന നമ്പറിൽ വിളിക്കുക.";
            default:
                return "I don't have reliable information to answer that. Please contact your local disaster management authority. In an emergency, dial 112.";
        }
    }
}
