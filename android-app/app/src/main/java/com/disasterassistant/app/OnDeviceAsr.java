package com.disasterassistant.app;

import android.content.Context;

import dev.ffmpegkit.whisper.Whisper;
import dev.ffmpegkit.whisper.WhisperConfig;
import dev.ffmpegkit.whisper.WhisperModel;
import dev.ffmpegkit.whisper.WhisperResult;

/**
 * On-device speech-to-text via whisper-android
 * (dev.ffmpegkit-maintained:whisper-android), a prebuilt whisper.cpp AAR.
 * Loaded once at startup from a model file fetched through the same
 * SAF-picker / Wi-Fi-hotspot flow as the LLM (see MainActivity) --
 * ggml-small.bin (~465MB), not "tiny" or "base": those are the sizes the
 * library's own docs recommend for English, but accuracy on 5 Indian
 * languages matters more here than the usual speed/size trade-off, and
 * only the multilingual variants (not the ".en" ones) support them at
 * all.
 *
 * Whisper's Kotlin API is suspend-function based (see SuspendBridge for
 * how this calls it as a plain blocking Java method).
 *
 * UNVERIFIED AT RUNTIME: written and compiled against the library's
 * documented API and bytecode, but not yet run on a physical phone from
 * this environment (no device access here). Treat every behavioural
 * claim as untested until it's actually run -- see
 * docs/implementation_status.md.
 */
public class OnDeviceAsr {

    private final WhisperModel model;

    public OnDeviceAsr(Context context, String modelPath) throws Exception {
        this.model = SuspendBridge.await(continuation ->
                Whisper.INSTANCE.loadModel(context, modelPath, continuation));
    }

    /**
     * @param wavPath path to a WAV file -- any sample rate/channel count
     *                is fine, whisper-android decodes and resamples to
     *                16kHz mono internally (confirmed from the library's
     *                README, not assumed).
     * @return the recognized text, auto-detecting the spoken language
     *         (WhisperConfig's default is language="auto", no
     *         translation) so the result stays in its original script --
     *         the same script-based detection the rest of this app's
     *         on-device pipeline already uses (see
     *         JsBridge.detectLanguageFromScript) then works on it
     *         unchanged.
     */
    public String transcribe(String wavPath) throws Exception {
        WhisperResult result = SuspendBridge.await(continuation ->
                Whisper.INSTANCE.transcribe(model, wavPath, new WhisperConfig(), continuation));
        return result.getText();
    }

    /**
     * @param language ISO code the user chose ("ta", "te"...), or "auto"
     * @param threads  CPU threads for whisper.cpp
     */
    public String transcribe(String wavPath, String language, int threads) throws Exception {
        WhisperConfig config = new WhisperConfig(language, false, threads, 0, false);
        WhisperResult result = SuspendBridge.await(continuation ->
                Whisper.INSTANCE.transcribe(model, wavPath, config, continuation));
        return result.getText();
    }

    public void close() {
        Whisper.INSTANCE.releaseModel(model);
    }
}
