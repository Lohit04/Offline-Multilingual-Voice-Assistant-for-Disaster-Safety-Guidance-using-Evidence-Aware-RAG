package com.disasterassistant.app;

import android.content.Context;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Pure-Java BM25 lexical search over the bundled chunks.jsonl (the exact
 * same chunk data the server-side pipeline indexes, copied into
 * assets/data/ -- see scripts/build_android_assets.py). No embeddings, no
 * FAISS: there is no portable mobile build of either, and a phone-sized
 * corpus (1,322 chunks here) is small enough that lexical search alone is
 * fast and simple to get right in the time available. This means the
 * standalone app's retrieval quality is BM25-only, a real step down from
 * the server's hybrid BM25+dense retrieval -- an explicit, documented
 * trade-off, not an oversight.
 */
public class OnDeviceRetrieval {

    public static class Chunk {
        public String chunkId;
        public String documentId;
        public String disasterCategory;
        public int page;
        public String text;
        List<String> tokens;
    }

    public static class Result {
        public Chunk chunk;
        public double score;
    }

    private final List<Chunk> chunks = new ArrayList<>();
    private final Map<String, Integer> docFreq = new HashMap<>();
    private double avgDocLen;

    /** @param guideFloor minimum survival-guide score to answer from it (JsBridge's sufficiency floor). */
    public OnDeviceRetrieval(Context context, String assetPath, double guideFloor) throws Exception {
        this.guideFloor = guideFloor;
        InputStream is = context.getAssets().open(assetPath);
        BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8));
        String line;
        long totalLen = 0;
        while ((line = reader.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            JSONObject obj = new JSONObject(line);
            Chunk c = new Chunk();
            c.chunkId = obj.getString("chunk_id");
            c.documentId = obj.getString("document_id");
            c.disasterCategory = obj.getString("disaster_category");
            c.page = obj.getInt("page");
            c.text = obj.getString("text");
            c.tokens = lexicalTokens(c.text);
            totalLen += c.tokens.size();
            Set<String> seen = new HashSet<>(c.tokens);
            for (String t : seen) {
                docFreq.merge(t, 1, Integer::sum);
            }
            chunks.add(c);
        }
        reader.close();
        avgDocLen = chunks.isEmpty() ? 1.0 : (double) totalLen / chunks.size();
    }

    /** Whitespace-split + edge-punctuation-strip -- same approach as (and
     * for the same reason as) the server's Python tokenizer: a plain
     * [a-z0-9]+ regex, or a Unicode \w+ one, both mis-tokenize scripts that
     * use combining marks (see translation.py's docstring on this
     * project's own prior bug with that). Splitting on whitespace alone
     * never looks inside a non-space run, so it can't make that mistake. */
    static List<String> tokenize(String text) {
        List<String> out = new ArrayList<>();
        for (String raw : text.toLowerCase().split("\\s+")) {
            String t = stripEdgePunctuation(raw);
            if (!t.isEmpty()) out.add(t);
        }
        return out;
    }

    private static String stripEdgePunctuation(String s) {
        int start = 0, end = s.length();
        while (start < end && !Character.isLetterOrDigit(s.charAt(start))) start++;
        while (end > start && !Character.isLetterOrDigit(s.charAt(end - 1))) end--;
        return start < end ? s.substring(start, end) : "";
    }

    // Same list as the server's knowledge/indexing.py::_STOPWORDS. Without
    // it, "what should I do if my" dominated BM25 and a flood question
    // retrieved forest-fire/cyclone/earthquake passages (measured on the
    // bundled chunks, 2026-10-14: top-3 for "floodwater enters my house"
    // was forest_fire, cyclone, earthquake). "not"/"no" are deliberately
    // kept -- they carry meaning in safety text.
    private static final Set<String> STOPWORDS = new HashSet<>(java.util.Arrays.asList((
            "a an the and or but if of to in on at by for with from as is are was were be been being am "
            + "i me my we our you your he she it its they them their this that these those "
            + "what which who whom whose when where why how should would could can will shall may might must "
            + "do does did done have has had having there here than then so such into onto about over under "
            + "again also just very any some all each other").split(" ")));

    /** Mirrors indexing.py::_stem -- minimal plural stripping, ASCII only. */
    static String stem(String t) {
        if (t.length() <= 3 || !t.chars().allMatch(ch -> ch < 128 && Character.isLetter(ch))) return t;
        if (t.endsWith("ies")) return t.substring(0, t.length() - 3) + "y";
        if (t.endsWith("ss") || t.endsWith("us") || t.endsWith("is")) return t;
        return t.endsWith("s") ? t.substring(0, t.length() - 1) : t;
    }

    /** Mirrors indexing.py::lexical_tokens (index AND query side). */
    static List<String> lexicalTokens(String text) {
        List<String> out = new ArrayList<>();
        for (String t : tokenize(text)) {
            if (!STOPWORDS.contains(t)) out.add(stem(t));
        }
        return out;
    }

    // Hazard words (prefix-matched on lexical tokens) -> chunk categories.
    // If a question names a hazard, only that hazard's passages are
    // eligible: BM25 alone has no notion of topic, and mixing hazards in
    // the prompt is how "floodwater in my house" got fire advice.
    static final String[][] HAZARDS = {
            {"flood", "flood", "urban_flood"}, {"inundat", "flood", "urban_flood"}, {"waterlog", "urban_flood"},
            {"cyclone", "cyclone"}, {"hurricane", "cyclone"},
            {"earthquake", "earthquake"}, {"tremor", "earthquake"},
            {"landslide", "landslide"}, {"mudslide", "landslide"},
            {"tsunami", "tsunami"}, {"heat", "heat_wave"}, {"cold", "cold_wave"},
            {"forest", "forest_fire"}, {"wildfire", "forest_fire"}, {"fire", "fire", "forest_fire"},
            {"lightning", "lightning"}, {"thunder", "thunderstorm_dust_storm_squall", "lightning"},
            {"dust", "thunderstorm_dust_storm_squall"}, {"avalanche", "avalanche"}, {"drought", "drought"},
            {"chemical", "chemical_emergency"}, {"nuclear", "nuclear_radiological_emergency"},
            {"radiation", "nuclear_radiological_emergency"}, {"smog", "smog_air_pollution"},
            {"pollution", "smog_air_pollution"}, {"epidemic", "biological_emergency"},
            {"pandemic", "biological_emergency"},
    };

    static Set<String> hazardCategories(String query) {
        Set<String> out = new HashSet<>();
        for (String t : lexicalTokens(query)) {
            for (String[] h : HAZARDS) {
                if (t.startsWith(h[0])) {
                    for (int i = 1; i < h.length; i++) out.add(h[i]);
                }
            }
        }
        return out;
    }

    // The one citizen-facing document. The other three are NDMA
    // management guidelines written for officials (building codes, agency
    // roles); for "what do I do?" questions they ranked first with text
    // like "cement mortar of 1:4". Search the survival guide first and
    // fall back to the guidelines only if it has no sufficient match --
    // which still lets policy questions ("role of NDMA in flood
    // management") reach the guidelines.
    private static final String CITIZEN_GUIDE = "ndma_disaster_survival_guide_2020";
    private final double guideFloor;

    public List<Result> search(String query, int topK) {
        Set<String> hazards = hazardCategories(query);
        List<Result> guide = score(query, c -> (hazards.isEmpty() || hazards.contains(c.disasterCategory))
                && CITIZEN_GUIDE.equals(c.documentId));
        if (!guide.isEmpty() && guide.get(0).score >= guideFloor) {
            return guide.subList(0, Math.min(topK, guide.size()));
        }
        List<Result> all = score(query, c -> hazards.isEmpty() || hazards.contains(c.disasterCategory));
        return all.subList(0, Math.min(topK, all.size()));
    }

    private List<Result> score(String query, java.util.function.Predicate<Chunk> eligible) {
        List<String> qTokens = lexicalTokens(query);
        int n = Math.max(chunks.size(), 1);
        double k1 = 1.5, b = 0.75;

        List<Result> results = new ArrayList<>();
        for (Chunk c : chunks) {
            if (!eligible.test(c)) continue;
            Map<String, Integer> tf = new HashMap<>();
            for (String t : c.tokens) tf.merge(t, 1, Integer::sum);

            double score = 0;
            for (String qt : qTokens) {
                int df = docFreq.getOrDefault(qt, 0);
                if (df == 0) continue;
                double idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
                int f = tf.getOrDefault(qt, 0);
                double denom = f + k1 * (1 - b + b * (c.tokens.size() / avgDocLen));
                score += idf * (f * (k1 + 1)) / (denom == 0 ? 1 : denom);
            }
            if (score > 0) {
                Result r = new Result();
                r.chunk = c;
                r.score = score;
                results.add(r);
            }
        }
        results.sort((x, y) -> Double.compare(y.score, x.score));
        return results;
    }
}
