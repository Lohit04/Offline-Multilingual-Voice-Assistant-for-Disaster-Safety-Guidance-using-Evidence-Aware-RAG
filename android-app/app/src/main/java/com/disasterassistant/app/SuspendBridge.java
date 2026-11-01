package com.disasterassistant.app;

import kotlin.Result;
import kotlin.coroutines.Continuation;
import kotlin.coroutines.CoroutineContext;
import kotlin.coroutines.EmptyCoroutineContext;
import kotlin.coroutines.intrinsics.IntrinsicsKt;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Calls a Kotlin `suspend fun` from Java, blocking the calling thread
 * until it completes. Needed because whisper-android
 * (dev.ffmpegkit-maintained:whisper-android) exposes its whole API as
 * suspend functions (Whisper.loadModel, Whisper.loadModelFromAsset,
 * Whisper.transcribe) with no plain-blocking Java overload -- and this
 * project stayed Java-only rather than add the Kotlin Android plugin for
 * three call sites.
 *
 * How this works, verified directly against kotlin-stdlib 2.0.21's
 * compiled bytecode (javap) rather than assumed from a blog post:
 *  - A suspend fun compiles to a normal JVM method with one extra
 *    trailing Continuation<T> parameter, returning Object.
 *  - If it finishes synchronously, it returns the real result directly
 *    and never touches the continuation.
 *  - If it needs to suspend, it returns the sentinel
 *    IntrinsicsKt.getCOROUTINE_SUSPENDED() immediately, and later --
 *    from whatever thread finishes the work -- calls
 *    continuation.resumeWith(Object) exactly once.
 *  - kotlin.Result<T> is an inline class: resumeWith's argument is the
 *    raw success value as-is on success, or a kotlin.Result$Failure
 *    instance (with a public `exception` field) on failure. No special
 *    unboxing needed for the success case.
 * Only ever call this from a background thread -- it blocks on a latch.
 */
final class SuspendBridge {

    private SuspendBridge() {
    }

    interface SuspendCall<T> {
        Object invoke(Continuation<T> continuation);
    }

    static <T> T await(SuspendCall<T> call) throws Exception {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicReference<Object> resultRef = new AtomicReference<>();
        AtomicReference<Throwable> errorRef = new AtomicReference<>();

        Continuation<T> continuation = new Continuation<T>() {
            @Override
            public CoroutineContext getContext() {
                return EmptyCoroutineContext.INSTANCE;
            }

            @Override
            public void resumeWith(Object result) {
                if (result instanceof Result.Failure) {
                    errorRef.set(((Result.Failure) result).exception);
                } else {
                    resultRef.set(result);
                }
                latch.countDown();
            }
        };

        Object immediate = call.invoke(continuation);
        if (immediate != IntrinsicsKt.getCOROUTINE_SUSPENDED()) {
            // Completed without suspending -- resumeWith was never called,
            // "immediate" is the real result.
            @SuppressWarnings("unchecked")
            T value = (T) immediate;
            return value;
        }

        latch.await();
        Throwable error = errorRef.get();
        if (error != null) {
            if (error instanceof Exception) throw (Exception) error;
            throw new RuntimeException(error);
        }
        @SuppressWarnings("unchecked")
        T value = (T) resultRef.get();
        return value;
    }
}
