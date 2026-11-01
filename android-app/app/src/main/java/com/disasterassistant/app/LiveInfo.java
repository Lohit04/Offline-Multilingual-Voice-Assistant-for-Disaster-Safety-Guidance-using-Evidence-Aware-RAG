package com.disasterassistant.app;

import java.util.regex.Pattern;

/**
 * Detects questions asking for live, current information (shelter status,
 * road closures, today's weather, evacuation orders) that static documents
 * cannot answer. Port of the server's safety/live_info.py patterns, which
 * were widened during the 2026-10-13 multilingual evaluation; keep the two
 * lists in sync. English only, like the server's.
 */
final class LiveInfo {

    private static final Pattern LIVE = Pattern.compile(String.join("|",
            "\\bright now\\b",
            "\\bcurrently\\b",
            "\\bat this (moment|time)\\b",
            "\\b(is|are)\\s+\\w+\\s+open\\b",
            "\\bnearest (open|available)\\b",
            "\\bcurrent (location|status|condition|road|weather|situation)\\b",
            "\\bavailable (now|today)\\b",
            "\\blive (update|status)\\b",
            "\\bhas .* been evacuated\\b",
            "\\bwhich (shelters?|camps?|roads?) (is|are) (open|closed|blocked|available)\\b",
            "\\b(today|tonight|tomorrow)\\b",
            "\\bstill (closed|open|blocked|flooded|raining|available)\\b",
            "\\border(s|ed)?\\b.{0,40}\\bevacuat",
            "\\bevacuat\\w*\\b.{0,40}\\b(order|ordered|announced|issued)\\b",
            "\\b(road|roads|bridge|route|highway)\\b.{0,40}\\b(closed|blocked|open|flooded|washed)\\b",
            "\\bnearest (shelter|camp|hospital|relief)\\b",
            "\\b(will it rain|forecast|weather in)\\b",
            "\\bso far\\b",
            "\\bcurrent (flood|cyclone|earthquake|disaster|storm|emergency|alert|warning)s?\\b",
            "\\b(warning|alert)s?\\b.{0,40}\\b(now|today|currently|issued|in effect|active)\\b",
            "\\b(is|are) there (a|an|any) .{0,30}(warning|alert|evacuation)\\b",
            "\\b(how many|number of)\\b.{0,50}\\b(rescued|killed|affected|evacuated|died|injured|missing|dead)\\b",
            "\\b(is|are|has|have)\\b.{0,50}\\b(now|at the moment|at present)\\b"
    ), Pattern.CASE_INSENSITIVE);

    private LiveInfo() {
    }

    static boolean isLiveInfoQuery(String text) {
        return LIVE.matcher(text).find();
    }

    // "now / today / currently" in each Indian language -- same list as
    // live_info.py::LIVE_TIME_WORDS (machine-chosen, unreviewed). Used only to
    // keep such questions off the instant-answer path.
    private static final String[] TIME_WORDS = {
            "अभी", "आज", "वर्तमान", "फिलहाल", "इस समय",
            "இப்போது", "இன்று", "தற்போது", "இப்ப", "இந்த நேரத்தில்",
            "ఇప్పుడు", "ఈరోజు", "ఈ రోజు", "ప్రస్తుతం", "ఈ సమయంలో",
            "ಈಗ", "ಇಂದು", "ಪ್ರಸ್ತುತ", "ಈ ಸಮಯದಲ್ಲಿ",
            "ഇപ്പോൾ", "ഇന്ന്", "നിലവിൽ", "ഈ സമയത്ത്",
    };

    static boolean mentionsCurrentTime(String text) {
        if (isLiveInfoQuery(text)) return true;
        for (String w : TIME_WORDS) if (text.contains(w)) return true;
        return false;
    }
}
