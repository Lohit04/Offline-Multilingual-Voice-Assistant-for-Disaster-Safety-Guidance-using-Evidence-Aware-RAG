package com.disasterassistant.app;

import android.content.Context;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Instant answers on the phone: matches a question to a situation prepared
 * in advance (scripts/build_answer_cache.py, bundled as
 * assets/data/answer_cache.json) and returns its stored, cited answer with
 * no LLM call -- milliseconds instead of the ~20-30 s Gemma takes.
 *
 * Matching is character-3-gram TF-IDF cosine against every stored phrasing
 * in all six languages (the phone has no embedding model). Character
 * n-grams rather than words because Tamil/Telugu/Kannada/Malayalam attach
 * endings to words, so whole-word matching misses most paraphrases.
 *
 * This is an exact port of scripts/calibrate_phone_matcher.py, which
 * chose MIN_SCORE / MIN_MARGIN on the evaluation sets (not used to build
 * the cache) such that no off-topic or live-information question, and no
 * question about a different hazard, got an instant answer. Change one,
 * change both.
 */
public class InstantAnswers {

    // Two-step matching, as on the web: if the question names a hazard
    // (English keyword or the Indian-language glossary below) only that
    // hazard's situations are candidates; otherwise all are, with a stricter
    // bar. Calibrated by scripts/calibrate_phone_matcher.py -- see class doc.
    // Values from calibrate_phone_matcher.py (with a 0.05 safety buffer) and
    // probe_cache_rejects.py (2026-10-26): on the evaluation sets 80/126
    // answerable questions instant and correct, 0 wrong-hazard, 0
    // off-topic/live; on 18 off-topic hazard-mentioning probes, 1 leak
    // ("invest in property in a flood zone" -> flood safety steps).
    static final double MIN_SCORE_KNOWN = 0.32;
    static final double MIN_MARGIN_KNOWN = 0.0;
    static final double MIN_SCORE_ANY = 0.30;
    static final double MIN_MARGIN_ANY = 0.10;
    // Reject set (add_cache_rejects.py): "What causes earthquakes?", insurance,
    // history... A question must beat its closest reject phrasing by this much.
    static final double REJECT_MARGIN = 0.05;

    // translation.py::HAZARD_GLOSSARY (stems) mapped to cache categories via
    // answer_cache.py::GLOSSARY_TO_CATS. Keep in sync with both.
    private static final String[][] GLOSSARY = {
            // hi
            {"बाढ़", "flood", "urban_flood"}, {"चक्रवात", "cyclone"}, {"भूकंप", "earthquake"}, {"सुनामी", "tsunami"},
            {"भूस्खलन", "landslide"}, {"आकाशीय बिजली", "lightning", "thunderstorm_dust_storm_squall"}, {"लू", "heat_wave"},
            // ta
            {"வெள்ள", "flood", "urban_flood"}, {"புயல", "cyclone"}, {"நிலநடுக்க", "earthquake"}, {"சுனாமி", "tsunami"},
            {"நிலச்சரிவ", "landslide"}, {"மின்னல", "lightning", "thunderstorm_dust_storm_squall"}, {"வெப்ப அலை", "heat_wave"},
            // te
            {"వరద", "flood", "urban_flood"}, {"తుఫాన", "cyclone"}, {"భూకంప", "earthquake"}, {"సునామీ", "tsunami"},
            {"కొండచరియ", "landslide"}, {"పిడుగ", "lightning", "thunderstorm_dust_storm_squall"}, {"వడగాల", "heat_wave"},
            {"మంటల", "fire", "forest_fire"},
            // kn
            {"ಪ್ರವಾಹ", "flood", "urban_flood"}, {"ಚಂಡಮಾರುತ", "cyclone"}, {"ಭೂಕಂಪ", "earthquake"}, {"ಸುನಾಮಿ", "tsunami"},
            {"ಭೂಕುಸಿತ", "landslide"}, {"ಸಿಡಿಲ", "lightning", "thunderstorm_dust_storm_squall"}, {"ಬಿಸಿಗಾಳಿ", "heat_wave"},
            {"ಬೆಂಕಿ", "fire", "forest_fire"},
            // ml
            {"വെള്ളപ്പൊക്ക", "flood", "urban_flood"}, {"ചുഴലിക്കാറ്റ", "cyclone"}, {"ഭൂകമ്പ", "earthquake"},
            {"സുനാമി", "tsunami"}, {"മണ്ണിടിച്ച", "landslide"}, {"ഇടിമിന്ന", "lightning", "thunderstorm_dust_storm_squall"},
            {"ഉഷ്ണതരംഗ", "heat_wave"}, {"തീപിടിത്ത", "fire", "forest_fire"},
    };
    private static final int N = 3;

    public static class Match {
        public JSONObject intent;
        public double score;
        public double margin;
    }

    private final List<JSONObject> intents = new ArrayList<>();
    private final List<Map<String, Double>> vecs = new ArrayList<>();
    private final List<Map<String, Double>> rejectVecs = new ArrayList<>();
    private final List<Integer> owner = new ArrayList<>();
    private final Map<String, Double> idf = new HashMap<>();
    private double unseenIdf;

    public InstantAnswers(Context context, String assetPath) throws Exception {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(
                context.getAssets().open(assetPath), StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) sb.append(line).append('\n');
        }
        root = new JSONObject(sb.toString());
        JSONArray arr = root.getJSONArray("intents");
        for (int i = 0; i < arr.length(); i++) intents.add(arr.getJSONObject(i));
    }

    private final JSONObject root;
    private volatile boolean indexed = false;

    /**
     * Builds the matching vectors (~5 s on the test phone at start-up, while
     * the phone is busy opening the app). Called on a background thread right
     * after the page opens: the picture tiles only need the parsed answers
     * (byId), so they work at once; match() waits for this if it is still
     * running.
     */
    public synchronized void buildIndex() throws Exception {
        if (indexed) return;
        JSONArray arr = root.getJSONArray("intents");
        List<Map<String, Integer>> raw = new ArrayList<>();
        Map<String, Integer> df = new HashMap<>();
        for (int i = 0; i < arr.length(); i++) {
            JSONObject it = arr.getJSONObject(i);
            JSONObject qs = it.getJSONObject("questions");
            for (java.util.Iterator<String> langs = qs.keys(); langs.hasNext(); ) {
                JSONArray list = qs.getJSONArray(langs.next());
                for (int j = 0; j < list.length(); j++) {
                    Map<String, Integer> g = grams(list.getString(j));
                    raw.add(g);
                    owner.add(i);
                    for (String k : g.keySet()) df.merge(k, 1, Integer::sum);
                }
            }
        }
        int n = raw.size();
        for (Map.Entry<String, Integer> e : df.entrySet()) idf.put(e.getKey(), Math.log(1 + (double) n / e.getValue()));
        unseenIdf = Math.log(1 + n);
        for (Map<String, Integer> g : raw) vecs.add(weigh(g));
        JSONObject rq = root.optJSONObject("rejects") == null ? null
                : root.getJSONObject("rejects").optJSONObject("questions");
        if (rq != null) {
            for (java.util.Iterator<String> langs = rq.keys(); langs.hasNext(); ) {
                JSONArray list = rq.getJSONArray(langs.next());
                for (int j = 0; j < list.length(); j++) rejectVecs.add(weigh(grams(list.getString(j))));
            }
        }
        indexed = true;
    }

    /** The prepared situation with this id ("flood.during"), or null. */
    public JSONObject byId(String id) {
        for (JSONObject it : intents) if (id.equals(it.optString("id"))) return it;
        return null;
    }

    public JSONArray ids() {
        JSONArray a = new JSONArray();
        for (JSONObject it : intents) a.put(it.optString("id"));
        return a;
    }

    public int size() {
        return intents.size();
    }

    /** @return the matched situation, or null if nothing matches confidently. */
    public Match match(String question) {
        if (LiveInfo.mentionsCurrentTime(question)) return null;
        if (!indexed) {
            try {
                buildIndex();  // normally already done in the background
            } catch (Exception e) {
                return null;
            }
        }
        Set<String> allowed = isEnglish(question) ? englishHazards(question) : glossaryHazards(question);

        Map<String, Double> q = weigh(grams(question));
        Map<Integer, Double> bestPerIntent = new HashMap<>();
        for (int p = 0; p < vecs.size(); p++) {
            int o = owner.get(p);
            if (!allowed.isEmpty() && !allowed.contains(intents.get(o).optString("hazard"))) continue;
            Map<String, Double> v = vecs.get(p);
            double s = 0;
            for (Map.Entry<String, Double> e : q.entrySet()) {
                Double w = v.get(e.getKey());
                if (w != null) s += e.getValue() * w;
            }
            Double prev = bestPerIntent.get(o);
            if (prev == null || s > prev) bestPerIntent.put(o, s);
        }
        int top = -1;
        double topS = -1, second = 0;
        for (Map.Entry<Integer, Double> e : bestPerIntent.entrySet()) {
            if (e.getValue() > topS) {
                if (top != -1) second = Math.max(second, topS);
                top = e.getKey();
                topS = e.getValue();
            } else {
                second = Math.max(second, e.getValue());
            }
        }
        double minScore = allowed.isEmpty() ? MIN_SCORE_ANY : MIN_SCORE_KNOWN;
        double minMargin = allowed.isEmpty() ? MIN_MARGIN_ANY : MIN_MARGIN_KNOWN;
        if (top == -1 || topS < minScore || topS - second < minMargin) return null;
        double rejectBest = -1;
        for (Map<String, Double> v : rejectVecs) {
            double r = 0;
            for (Map.Entry<String, Double> e : q.entrySet()) {
                Double w = v.get(e.getKey());
                if (w != null) r += e.getValue() * w;
            }
            if (r > rejectBest) rejectBest = r;
        }
        if (rejectBest >= topS - REJECT_MARGIN) return null;
        Match m = new Match();
        m.intent = intents.get(top);
        m.score = topS;
        m.margin = topS - second;
        return m;
    }

    // --- must match calibrate_phone_matcher.py exactly ---

    static String normalise(String text) {
        StringBuilder sb = new StringBuilder();
        String lower = text.toLowerCase(Locale.ROOT);
        for (int i = 0; i < lower.length(); i++) {
            char c = lower.charAt(i);
            sb.append(keep(c) ? c : ' ');
        }
        return " " + sb.toString().trim().replaceAll("\\s+", " ") + " ";
    }

    // Letters, numbers AND combining marks: Indic vowel signs are marks, so
    // Character.isLetterOrDigit alone would cut every Tamil word apart.
    private static boolean keep(char c) {
        switch (Character.getType(c)) {
            case Character.UPPERCASE_LETTER: case Character.LOWERCASE_LETTER: case Character.TITLECASE_LETTER:
            case Character.MODIFIER_LETTER: case Character.OTHER_LETTER:
            case Character.DECIMAL_DIGIT_NUMBER: case Character.LETTER_NUMBER: case Character.OTHER_NUMBER:
            case Character.NON_SPACING_MARK: case Character.ENCLOSING_MARK: case Character.COMBINING_SPACING_MARK:
                return true;
            default:
                return false;
        }
    }

    static Map<String, Integer> grams(String text) {
        String t = normalise(text);
        Map<String, Integer> g = new HashMap<>();
        for (int i = 0; i + N <= t.length(); i++) g.merge(t.substring(i, i + N), 1, Integer::sum);
        return g;
    }

    private Map<String, Double> weigh(Map<String, Integer> g) {
        Map<String, Double> v = new HashMap<>();
        double norm = 0;
        for (Map.Entry<String, Integer> e : g.entrySet()) {
            double w = e.getValue() * idf.getOrDefault(e.getKey(), unseenIdf);
            v.put(e.getKey(), w);
            norm += w * w;
        }
        norm = Math.sqrt(norm);
        if (norm == 0) norm = 1;
        for (Map.Entry<String, Double> e : v.entrySet()) e.setValue(e.getValue() / norm);
        return v;
    }

    static boolean isEnglish(String q) {
        for (int i = 0; i < q.length(); i++) if (q.charAt(i) > 0x024F) return false;
        return true;
    }

    static Set<String> glossaryHazards(String q) {
        Set<String> out = new HashSet<>();
        for (String[] g : GLOSSARY) {
            if (q.contains(g[0])) {
                for (int i = 1; i < g.length; i++) out.add(g[i]);
            }
        }
        return out;
    }

    private static final Pattern WORD = Pattern.compile("[a-z]+");

    static Set<String> englishHazards(String q) {
        Set<String> out = new HashSet<>();
        Matcher m = WORD.matcher(q.toLowerCase(Locale.ROOT));
        while (m.find()) {
            String w = m.group();
            for (String[] h : OnDeviceRetrieval.HAZARDS) {
                if (w.startsWith(h[0])) {
                    for (int i = 1; i < h.length; i++) out.add(h[i]);
                }
            }
        }
        return out;
    }
}
