package com.disasterassistant.app;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Minimal 16kHz mono 16-bit PCM WAV recorder using AudioRecord directly.
 *
 * Deliberately NOT the WebView's getUserMedia()/MediaRecorder (what the
 * network-connected build and app.js's non-on-device path both use) --
 * that produces a compressed WebM/Opus blob, and whisper-android (see
 * OnDeviceAsr) decodes WAV/MP3/FLAC, not WebM. Recording natively also
 * means this path never touches a browser audio API or the network,
 * matching the rest of the on-device pipeline.
 */
public class WavRecorder {

    private static final int SAMPLE_RATE = 16000;

    private AudioRecord audioRecord;
    private Thread recordingThread;
    private final AtomicBoolean recording = new AtomicBoolean(false);
    private File outputFile;
    private File rawPcmFile;

    public void start(File outputFile) throws IOException {
        this.outputFile = outputFile;
        int minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (minBufferSize <= 0) {
            throw new IOException("This device does not support 16kHz mono recording");
        }

        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, minBufferSize * 4);
        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            audioRecord.release();
            audioRecord = null;
            throw new IOException("Could not initialize the microphone (missing permission?)");
        }

        rawPcmFile = new File(outputFile.getParentFile(), outputFile.getName() + ".pcm");
        recording.set(true);
        audioRecord.startRecording();

        final int bufferSize = minBufferSize;
        recordingThread = new Thread(() -> {
            try (FileOutputStream out = new FileOutputStream(rawPcmFile)) {
                byte[] buffer = new byte[bufferSize];
                while (recording.get()) {
                    int read = audioRecord.read(buffer, 0, buffer.length);
                    if (read > 0) out.write(buffer, 0, read);
                }
            } catch (IOException ignored) {
                // Nothing to do if the write fails mid-recording; stop()
                // will still produce whatever was captured so far.
            }
        }, "wav-recorder");
        recordingThread.start();
    }

    /**
     * @return the finished WAV file (same File passed to start()).
     */
    public File stop() throws IOException, InterruptedException {
        recording.set(false);
        if (audioRecord != null) {
            audioRecord.stop();
            audioRecord.release();
            audioRecord = null;
        }
        if (recordingThread != null) {
            recordingThread.join();
        }
        writeWavFile(rawPcmFile, outputFile, SAMPLE_RATE);
        rawPcmFile.delete();
        return outputFile;
    }

    private static void writeWavFile(File pcmFile, File wavFile, int sampleRate) throws IOException {
        long pcmLen = pcmFile.length();
        int channels = 1;
        int bitsPerSample = 16;
        int byteRate = sampleRate * channels * bitsPerSample / 8;

        try (FileOutputStream out = new FileOutputStream(wavFile)) {
            byte[] header = new byte[44];
            writeAscii(header, 0, "RIFF");
            writeInt32(header, 4, (int) (pcmLen + 36));
            writeAscii(header, 8, "WAVE");
            writeAscii(header, 12, "fmt ");
            writeInt32(header, 16, 16);
            writeInt16(header, 20, (short) 1); // PCM
            writeInt16(header, 22, (short) channels);
            writeInt32(header, 24, sampleRate);
            writeInt32(header, 28, byteRate);
            writeInt16(header, 32, (short) (channels * bitsPerSample / 8));
            writeInt16(header, 34, (short) bitsPerSample);
            writeAscii(header, 36, "data");
            writeInt32(header, 40, (int) pcmLen);
            out.write(header);

            try (FileInputStream in = new FileInputStream(pcmFile)) {
                byte[] buf = new byte[8192];
                int n;
                while ((n = in.read(buf)) != -1) {
                    out.write(buf, 0, n);
                }
            }
        }
    }

    private static void writeAscii(byte[] data, int offset, String value) {
        for (int i = 0; i < value.length(); i++) {
            data[offset + i] = (byte) value.charAt(i);
        }
    }

    private static void writeInt32(byte[] data, int offset, int value) {
        data[offset] = (byte) (value & 0xff);
        data[offset + 1] = (byte) ((value >> 8) & 0xff);
        data[offset + 2] = (byte) ((value >> 16) & 0xff);
        data[offset + 3] = (byte) ((value >> 24) & 0xff);
    }

    private static void writeInt16(byte[] data, int offset, short value) {
        data[offset] = (byte) (value & 0xff);
        data[offset + 1] = (byte) ((value >> 8) & 0xff);
    }
}
