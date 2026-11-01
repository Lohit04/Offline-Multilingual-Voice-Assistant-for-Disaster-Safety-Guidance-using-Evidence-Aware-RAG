package com.disasterassistant.app;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.net.Uri;
import android.net.http.SslError;
import android.os.Bundle;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.View;
import android.webkit.PermissionRequest;
import android.webkit.SslErrorHandler;
import android.webkit.WebChromeClient;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Standalone, on-device mode: an LLM (MediaPipe + Qwen2.5-0.5B-Instruct),
 * speech-to-text (whisper-android / whisper.cpp), BM25 retrieval over the
 * bundled knowledge-base chunks, and Android's built-in TextToSpeech all
 * run ON THE PHONE -- no server, no network, once the two one-time model
 * downloads below have happened. See JsBridge.java for exactly what's
 * scoped down compared to the server pipeline (BM25 only, no live-info
 * guard, a much smaller LLM) -- this is a real, documented trade-off for
 * getting something running fully standalone, not a hidden one.
 *
 * The UI itself is the same bundled HTML/CSS/JS as the network-connected
 * build (assets/www/) -- app.js detects AndroidBridge's presence and
 * routes through it instead of fetch() when available.
 */
public class MainActivity extends Activity {

    private static final String LOCAL_PAGE = "file:///android_asset/www/index.html";
    private static final int MIC_PERMISSION_REQUEST = 1;
    private static final int SELECT_MODEL_REQUEST = 2;

    /** One model file this app needs once, with where to get it. */
    private static final class ModelSpec {
        final String filename;
        final long expectedSize;
        final String humanName;
        final String downloadPageUrl;
        final String hotspotUrl;

        ModelSpec(String filename, long expectedSize, String humanName,
                  String downloadPageUrl, String hotspotUrl) {
            this.filename = filename;
            this.expectedSize = expectedSize;
            this.humanName = humanName;
            this.downloadPageUrl = downloadPageUrl;
            this.hotspotUrl = hotspotUrl;
        }
    }

    // Gemma-3 1B int4: the model Google's own Android LLM Inference guide
    // uses. Replaced Qwen2.5-0.5B (litert-community "multi-prefill-seq"
    // .task), which on a real device (vivo iQOO Z3, 2026-10-14) failed
    // every generation with "tflite: Input tensor 175 lacks data" under
    // both MediaPipe 0.10.27 and 0.10.35, CPU and GPU, any prompt length --
    // a model/runtime incompatibility, not fixable from this app.
    // Gated on Hugging Face: the user must log in and accept the Gemma
    // license once before the download link works. Size verified from the
    // user's actual download (554661243 bytes, valid .task zip header).
    private static final ModelSpec LLM_MODEL = new ModelSpec(
            "gemma3-1b-it-int4.task", 554661243L,
            "the on-device AI model (Gemma-3 1B, ~530MB; log in to Hugging Face and accept "
                    + "the Gemma license first)",
            "https://huggingface.co/litert-community/Gemma3-1B-IT/resolve/main/gemma3-1b-it-int4.task",
            "http://192.168.137.1:8080/gemma3-1b-it-int4.task");

    // Source: ggerganov/whisper.cpp on Hugging Face, same public-no-auth
    // verification. "small" (not "tiny"/"base") specifically because
    // speech-recognition accuracy across 5 Indian languages matters more
    // here than the speed/size trade-off those smaller sizes are tuned
    // for -- and only the multilingual variants (no ".en" suffix)
    // support non-English languages at all.
    //
    // 2026-10-14: switched small -> base for speed. On the test phone
    // (Snapdragon 768G, 5.3GB RAM, nearly all in use) "small" spent ~76 s
    // of CPU transcribing one short question, which alone broke the 15 s
    // target. "base" (148MB) is ~3x faster but less accurate, especially
    // for Tamil/Telugu/Kannada/Malayalam -- revisit before demoing those.
    //
    // 2026-10-26: base -> tiny. English and Hindi now use Android's own
    // on-device recognizer, so Whisper only serves Tamil. On the test phone,
    // base took 72 s for a 2.5 s English clip and 138 s for Telugu (still in
    // the wrong script); tiny, told "Tamil", took 4.2 s with 90% character
    // similarity. Telugu/Kannada/Malayalam: no offline engine produced usable
    // text on this phone (Whisper wrote romanised text; Google's recognizer
    // needs internet for them; Vosk has no Tamil and an 88% WER Telugu model),
    // so the page asks those users to type instead.
    private static final ModelSpec ASR_MODEL = new ModelSpec(
            "ggml-tiny.bin", 77691713L,
            "the offline speech-recognition model (~75MB)",
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
            "http://192.168.137.1:8080/ggml-tiny.bin");

    private static final ModelSpec[] REQUIRED_MODELS = {LLM_MODEL, ASR_MODEL};

    private WebView webView;
    private View setupLayout;
    private ProgressBar setupProgress;
    private TextView setupText;
    private Button selectModelBtn;
    private Button wifiDownloadBtn;
    private ExecutorService executor;
    static long createdAt;  // start-up timing reference (JsBridge.startupTimings)
    private volatile OnDeviceLlm llm;  // set by the background loader thread
    private JsBridge bridge;
    private TextToSpeech tts;

    // Index into REQUIRED_MODELS of the model currently being waited on.
    // Only touched from the single-threaded `executor` or in direct
    // response to a UI action on it, so no synchronization needed.
    private int pendingModelIndex = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        createdAt = android.os.SystemClock.elapsedRealtime();
        setContentView(R.layout.activity_main);
        // System bars in the app's own background colour: reads as an app,
        // not a browser tab.
        getWindow().setStatusBarColor(0xFF0B1020);
        getWindow().setNavigationBarColor(0xFF0B1020);

        webView = findViewById(R.id.webview);
        webView.setBackgroundColor(0xFF0B1020);  // no white flash before the page paints
        // Debug builds only: lets the page be driven and timed over USB
        // (Chrome DevTools protocol) for automated on-device testing.
        if ((getApplicationInfo().flags & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0) {
            WebView.setWebContentsDebuggingEnabled(true);
        }
        setupLayout = findViewById(R.id.setup_layout);
        setupProgress = findViewById(R.id.setup_progress);
        setupText = findViewById(R.id.setup_text);
        // The shield gently "breathes" while the app gets ready.
        View mark = findViewById(R.id.splash_mark);
        android.animation.ObjectAnimator pulse = android.animation.ObjectAnimator.ofPropertyValuesHolder(mark,
                android.animation.PropertyValuesHolder.ofFloat(View.SCALE_X, 1f, 1.08f),
                android.animation.PropertyValuesHolder.ofFloat(View.SCALE_Y, 1f, 1.08f));
        pulse.setDuration(900);
        pulse.setRepeatCount(android.animation.ValueAnimator.INFINITE);
        pulse.setRepeatMode(android.animation.ValueAnimator.REVERSE);
        pulse.start();
        selectModelBtn = findViewById(R.id.select_model_btn);
        wifiDownloadBtn = findViewById(R.id.wifi_download_btn);
        executor = Executors.newSingleThreadExecutor();

        selectModelBtn.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.setType("*/*");
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            startActivityForResult(intent, SELECT_MODEL_REQUEST);
        });
        wifiDownloadBtn.setOnClickListener(v -> downloadCurrentModelOverWifi());

        WebSettings settings = webView.getSettings();
        settings.setJavaScriptEnabled(true);
        settings.setDomStorageEnabled(true);
        settings.setMediaPlaybackRequiresUserGesture(false);
        settings.setAllowFileAccess(true);

        // The network-connected build (see docs/architecture.md) needed this
        // override because it talks to a server with a self-signed cert.
        // The standalone build makes no HTTPS calls at all, but this is
        // harmless to leave in and avoids a separate code path.
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onReceivedSslError(WebView view, SslErrorHandler handler, SslError error) {
                handler.proceed();
            }
        });

        webView.setWebChromeClient(new WebChromeClient() {
            @Override
            public void onPermissionRequest(final PermissionRequest request) {
                runOnUiThread(() -> request.grant(request.getResources()));
            }
        });

        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, MIC_PERMISSION_REQUEST);
        }

        // Warm the speech engine as soon as it's ready: rendering one word to
        // a throwaway file loads the voice, so the first real answer starts
        // speaking without that delay.
        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                tts.synthesizeToFile("Ready", null, new File(getCacheDir(), "tts_warmup.wav"), "warmup");
            }
        });

        setupOnDevicePipeline();
    }

    // Entry point: walks REQUIRED_MODELS in order, checking local storage
    // only -- never attempts a network call on its own. An unreachable
    // hotspot used to throw here, land in a catch block that never called
    // loadUrl(), and leave a dead setup screen with no way forward except
    // reinstalling. Both real acquisition paths now live in their own
    // methods below, triggered only by a button tap, and each reports its
    // own failure without losing the ability to retry.
    private void setupOnDevicePipeline() {
        executor.execute(this::checkNextModel);
    }

    private void checkNextModel() {
        while (pendingModelIndex < REQUIRED_MODELS.length) {
            ModelSpec spec = REQUIRED_MODELS[pendingModelIndex];
            File f = new File(getFilesDir(), spec.filename);
            if (f.exists() && f.length() == spec.expectedSize) {
                pendingModelIndex++;
                continue;
            }
            runOnUiThread(() -> {
                setupText.setText(
                        "Need " + spec.humanName + " (one-time download).\n\n" +
                        "Easiest: open this link in your phone's normal browser (any Wi-Fi or " +
                        "mobile data -- not just this app's hotspot) to download it once, then " +
                        "come back here and tap “Select the file I downloaded” below:\n\n" +
                        spec.downloadPageUrl);
                selectModelBtn.setVisibility(View.VISIBLE);
                wifiDownloadBtn.setVisibility(View.VISIBLE);
            });
            return;
        }
        finishSetup();
    }

    // Path 1 (recommended): the user already downloaded the current
    // pending model themselves, anywhere the system's document picker can
    // see it (typically Downloads). Uses Storage Access Framework
    // (ACTION_OPEN_DOCUMENT) specifically because it needs zero storage
    // permissions and is scoped-storage-safe on every API level this app
    // supports. The picked file is copied once into this app's own
    // private storage so every later launch finds it there.
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != SELECT_MODEL_REQUEST || resultCode != RESULT_OK || data == null) {
            return;
        }
        Uri uri = data.getData();
        if (uri == null) return;
        if (pendingModelIndex >= REQUIRED_MODELS.length) return;
        ModelSpec spec = REQUIRED_MODELS[pendingModelIndex];

        runOnUiThread(() -> {
            selectModelBtn.setVisibility(View.GONE);
            wifiDownloadBtn.setVisibility(View.GONE);
            setupText.setText("Copying " + spec.humanName + "...");
        });
        executor.execute(() -> {
            File dest = new File(getFilesDir(), spec.filename);
            try {
                try (InputStream in = getContentResolver().openInputStream(uri);
                     OutputStream out = new FileOutputStream(dest)) {
                    if (in == null) throw new IOException("Could not open the selected file");
                    byte[] buffer = new byte[65536];
                    int read;
                    while ((read = in.read(buffer)) != -1) {
                        out.write(buffer, 0, read);
                    }
                }
                if (dest.length() != spec.expectedSize) {
                    dest.delete();
                    throw new IOException("That file is " + dest.length() + " bytes; the correct " +
                            "file is exactly " + spec.expectedSize + " bytes. Make sure you picked " +
                            "the file downloaded from the link above, not a different one.");
                }
                pendingModelIndex++;
                checkNextModel();
            } catch (Exception e) {
                runOnUiThread(() -> {
                    setupText.setText("Could not use that file: " + e.getMessage());
                    selectModelBtn.setVisibility(View.VISIBLE);
                    wifiDownloadBtn.setVisibility(View.VISIBLE);
                });
            }
        });
    }

    // Path 2 (fallback): the original behaviour, kept for whoever would
    // rather not use a browser -- downloads the current pending model from
    // this project's own plain-HTTP server over this app's Wi-Fi hotspot.
    private void downloadCurrentModelOverWifi() {
        if (pendingModelIndex >= REQUIRED_MODELS.length) return;
        ModelSpec spec = REQUIRED_MODELS[pendingModelIndex];

        runOnUiThread(() -> {
            selectModelBtn.setVisibility(View.GONE);
            wifiDownloadBtn.setVisibility(View.GONE);
            setupText.setText("Downloading " + spec.humanName + " over this app's Wi-Fi hotspot...");
        });
        executor.execute(() -> {
            try {
                ModelDownloader.downloadIfMissing(this, spec.hotspotUrl, spec.filename, spec.expectedSize,
                        (downloaded, total) -> {
                            int pct = total > 0 ? (int) (downloaded * 100 / total) : 0;
                            runOnUiThread(() -> { setupProgress.setIndeterminate(false); setupProgress.setProgress(pct); });
                        });
                pendingModelIndex++;
                checkNextModel();
            } catch (Exception e) {
                runOnUiThread(() -> {
                    setupText.setText("Wi-Fi hotspot download failed: " + e.getMessage() +
                            "\n\nMake sure this app's hotspot/server is on and connected, or use the " +
                            "browser-download option instead.");
                    selectModelBtn.setVisibility(View.VISIBLE);
                    wifiDownloadBtn.setVisibility(View.VISIBLE);
                });
            }
        });
    }

    // Common tail once every REQUIRED_MODELS entry is present: load the
    // LLM, load retrieval, attach the JS bridge, and switch the WebView
    // over to the bundled UI. Speech-to-text is NOT loaded here -- JsBridge
    // loads it on the first voice question. Both models resident at once
    // is ~1GB of weights alone, and the test phone (5.3GB total) had only
    // ~1.3GB available; text questions shouldn't pay for a model they
    // never use.
    private void finishSetup() {
        try {
            File llmFile = new File(getFilesDir(), LLM_MODEL.filename);
            File asrFile = new File(getFilesDir(), ASR_MODEL.filename);

            // Instant answers first (well under a second): with them the page is
            // useful at once -- tiles, prepared answers, voice for common questions.
            runOnUiThread(() -> setupText.setText("Opening..."));
            InstantAnswers instant = null;
            long ti = android.os.SystemClock.elapsedRealtime();
            JsBridge.mark("setup_start_ms", ti - createdAt);
            try {
                instant = new InstantAnswers(this, "data/answer_cache.json");
                JsBridge.mark("instant_parse_ms", android.os.SystemClock.elapsedRealtime() - ti);
            } catch (Exception e) {
                // Not fatal: every question then goes through the live model.
                Log.w("DisasterTiming", "instant answers unavailable: " + e.getMessage());
            }
            bridge = new JsBridge(this, webView, null, asrFile.getAbsolutePath(), null, instant, tts, executor);
            final InstantAnswers ia = instant;
            if (ia != null) {
                new Thread(() -> {
                    long tb = android.os.SystemClock.elapsedRealtime();
                    try {
                        ia.buildIndex();
                    } catch (Exception e) {
                        Log.w("DisasterTiming", "instant index failed: " + e.getMessage());
                    }
                    JsBridge.mark("instant_vectors_ms", android.os.SystemClock.elapsedRealtime() - tb);
                }, "instant-index").start();
            }
            // Speech-to-text is not preloaded: English/Hindi use Android's own
            // recognizer and the Indian languages it can't do are typed, so
            // Whisper is only a lazy fallback (~100 MB of RAM otherwise, which
            // makes vivo's power manager likelier to kill the app).
            // One-time, needs internet: offline English (India) + Hindi speech packs.
            bridge.requestSpeechPacks();
            JsBridge.mark("bridge_ready_ms", android.os.SystemClock.elapsedRealtime() - createdAt);
            runOnUiThread(() -> {
                webView.addJavascriptInterface(bridge, "AndroidBridge");
                webView.loadUrl(LOCAL_PAGE);
                // Fade the start-up screen away once the page has had a moment to paint.
                setupLayout.animate().alpha(0f).setStartDelay(250).setDuration(300)
                        .withEndAction(() -> setupLayout.setVisibility(View.GONE)).start();
            });

            // Then the model and the search index (~12 s on the test phone), in
            // the background; questions that need them wait (JsBridge.enginesReady).
            // Started only once the page has drawn (app.js calls pageReady()):
            // started together, the model's GPU set-up stalled the first paint
            // by several seconds. Fallback timer in case the page never calls.
            final JsBridge b = bridge;
            b.setOnPageReady(() -> new Thread(() -> {
                try {
                    llm = new OnDeviceLlm(this, llmFile.getAbsolutePath());
                    OnDeviceRetrieval retrieval = new OnDeviceRetrieval(this, "data/chunks.jsonl", JsBridge.BM25_SUFFICIENCY_FLOOR);
                    b.setEngines(llm, retrieval);
                } catch (Exception e) {
                    Log.w("DisasterTiming", "model load failed: " + e.getMessage());
                    b.enginesFailed();
                }
            }, "engine-loader").start());
            webView.postDelayed(b::pageReady, 6000);
        } catch (Exception e) {
            runOnUiThread(() -> setupText.setText("Setup failed while loading a model: " + e.getMessage()));
        }
    }

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (llm != null) llm.close();
        if (bridge != null) bridge.close();
        if (tts != null) tts.shutdown();
        executor.shutdown();
    }
}
