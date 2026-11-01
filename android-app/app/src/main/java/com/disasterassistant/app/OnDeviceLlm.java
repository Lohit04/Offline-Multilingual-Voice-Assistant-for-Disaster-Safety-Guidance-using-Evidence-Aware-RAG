package com.disasterassistant.app;

import android.content.Context;
import android.util.Log;

import com.google.mediapipe.tasks.genai.llminference.LlmInference;

/**
 * Wraps MediaPipe's LlmInference task around a locally-stored Qwen2.5-0.5B-
 * Instruct .task file (chosen specifically because it's openly downloadable
 * without any Hugging Face license-gate click-through, unlike the Gemma
 * models MediaPipe's own docs lead with -- see
 * docs/implementation_status.md for the full story on why Gemma was ruled
 * out). 0.5B parameters is small/fast enough for a phone, at a real cost
 * to answer quality compared to the server's 8B model -- another explicit
 * trade-off of going standalone, not something to pretend is equivalent.
 */
public class OnDeviceLlm {

    private final LlmInference llmInference;

    // The bundled model file is named "..._ekv1280.task" -- it was exported
    // with a FIXED 1280-token KV cache (input + output tokens combined,
    // one shared budget). Exceeding that isn't a normal error: MediaPipe's
    // native inference engine overruns that fixed-size cache and crashes
    // the whole app process, not something a Java try/catch can stop (see
    // JsBridge.groundedPrompt's token budget, which this bounds exists to
    // backstop). Set comfortably under 1280, not at it, so there's margin
    // for tokenization differing slightly from a word/character estimate.
    private static final int MAX_TOKENS = 1024;

    /** Which backend actually loaded -- "GPU" or "CPU" -- for timing logs. */
    public final String backend;

    // GPU first, CPU fallback. History: an earlier crash was wrongly
    // blamed on the GPU (a generic "eglSurfaceAttribMTK" EGL message sat
    // next to it); the real cause turned out to be the Qwen .task file
    // ("Input tensor 175 lacks data" on CPU too). The test phone is a
    // Snapdragon 768G / Adreno 620, which MediaPipe's GPU path supports.
    // Only a failure to CREATE the GPU engine can be caught here; a failure
    // during inference would still abort the process (MediaPipe's JNI
    // layer calls NewByteArray with an exception pending), so GPU has to
    // be proven on a real device, not assumed.
    public OnDeviceLlm(Context context, String modelPath) {
        LlmInference engine;
        String chosen;
        try {
            engine = LlmInference.createFromOptions(context, options(modelPath, LlmInference.Backend.GPU));
            chosen = "GPU";
        } catch (Exception e) {
            Log.w(TAG, "GPU engine failed to load, falling back to CPU: " + e.getMessage());
            engine = LlmInference.createFromOptions(context, options(modelPath, LlmInference.Backend.CPU));
            chosen = "CPU";
        }
        this.llmInference = engine;
        this.backend = chosen;
        Log.w(TAG, "LLM loaded on " + chosen);
    }

    private static LlmInference.LlmInferenceOptions options(String modelPath, LlmInference.Backend backend) {
        return LlmInference.LlmInferenceOptions.builder()
                .setModelPath(modelPath)
                .setMaxTokens(MAX_TOKENS)
                .setMaxTopK(64)
                .setPreferredBackend(backend)
                .build();
    }

    public interface PartialListener {
        void onPartial(String textSoFar);
    }

    private static final String TAG = "DisasterLlm";

    public String generate(String prompt) {
        return llmInference.generateResponse(prompt);
    }

    /**
     * Same as generate(), but reports the text as it is produced so the UI
     * can show words immediately instead of after the whole answer. Blocks
     * until generation finishes and returns the full text.
     */
    public String generateStreaming(String prompt, PartialListener listener) throws Exception {
        StringBuffer soFar = new StringBuffer();
        String full = llmInference.generateResponseAsync(prompt, (partial, done) -> {
            if (partial != null) soFar.append(partial);
            if (listener != null) listener.onPartial(soFar.toString());
        }).get();
        return full != null && !full.isEmpty() ? full : soFar.toString();
    }

    public int sizeInTokens(String text) {
        try {
            return llmInference.sizeInTokens(text);
        } catch (Exception e) {
            return -1;
        }
    }

    public void close() {
        llmInference.close();
    }
}
