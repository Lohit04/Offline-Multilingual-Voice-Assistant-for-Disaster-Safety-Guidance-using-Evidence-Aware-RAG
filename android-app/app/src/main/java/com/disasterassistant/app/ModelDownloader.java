package com.disasterassistant.app;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * One-time download of the on-device model into this app's own private
 * storage (getFilesDir()) -- deliberately NOT the shared Downloads folder,
 * which would need Storage Access Framework / scoped-storage handling on
 * modern Android. Downloading straight into the app's sandbox avoids that
 * entirely. Source is this project's own plain-HTTP download server (see
 * server.py) -- the laptop needs to be running it for this ONE download;
 * every actual question afterward needs neither the laptop nor the
 * network.
 */
public class ModelDownloader {

    public interface ProgressListener {
        void onProgress(long downloaded, long total);
    }

    public static File downloadIfMissing(Context context, String url, String fileName, long expectedSize,
                                          ProgressListener listener) throws IOException {
        File dest = new File(context.getFilesDir(), fileName);
        if (dest.exists() && dest.length() == expectedSize) {
            return dest;
        }

        File tmp = new File(context.getFilesDir(), fileName + ".part");
        HttpURLConnection conn = (HttpURLConnection) new URL(url).openConnection();
        conn.setConnectTimeout(15000);
        conn.setReadTimeout(30000);
        conn.connect();
        long total = conn.getContentLength();

        try (InputStream in = conn.getInputStream(); OutputStream out = new FileOutputStream(tmp)) {
            byte[] buffer = new byte[65536];
            long downloaded = 0;
            int read;
            while ((read = in.read(buffer)) != -1) {
                out.write(buffer, 0, read);
                downloaded += read;
                if (listener != null) listener.onProgress(downloaded, total);
            }
        } finally {
            conn.disconnect();
        }

        if (!tmp.renameTo(dest)) {
            throw new IOException("Could not finalize downloaded file " + fileName);
        }
        return dest;
    }
}
