"""Ties retrieval -> evidence assessment -> grounded generation / abstention together."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from disaster_assistant import config
from disaster_assistant.retrieval.hybrid import HybridRetriever, RetrievedChunk
from disaster_assistant.safety.abstention import assess, EvidenceAssessment
from disaster_assistant.safety.live_info import apply_live_info_guard, mentions_current_time
from disaster_assistant.safety.evidence_check import evidence_supported
from disaster_assistant.safety.query_check import is_too_vague, CLARIFY_MESSAGE
from disaster_assistant.generation.prompts import build_grounded_prompt
from disaster_assistant.generation.translation import (
    translate_to_english, detect_language_and_translate, detect_language_from_script,
)
from disaster_assistant.retrieval.answer_cache import get_answer_cache
from disaster_assistant.generation import ollama_client


def _load_verified_phone_numbers() -> list[str]:
    """Pulls phone numbers straight from documents.json's verified_contacts
    (see that file for provenance/caveats on each number) instead of
    hardcoding them a second time here."""
    registry = json.loads(config.DOCUMENTS_REGISTRY.read_text(encoding="utf-8"))
    return [c["phone"] for c in registry.get("verified_contacts", [])]


_PHONE_NUMBERS = _load_verified_phone_numbers()
_CONTACTS_LINE_EN = "In an immediate emergency in India, dial " + " or ".join(_PHONE_NUMBERS) + "."
_CONTACTS_LINE_TA = "உடனடி அவசரநிலையில் " + " அல்லது ".join(_PHONE_NUMBERS) + " ஐ அழைக்கவும்."

# All non-English messages below are MACHINE TRANSLATIONS (Tamil: hand
# -written then unreviewed; Hindi/Telugu/Kannada/Malayalam: generated via
# the project's own Ollama model, 2026-10-10, manually checked only for
# "contains the right phone numbers, looks like the right script" -- NOT
# reviewed by a native speaker of any of these languages). Per CLAUDE.md
# SS12, do not present these as official/verified translations.
ABSTENTION_MESSAGE = {
    "en": (
        "I don't have reliable information to answer that from my local disaster-safety guidance. "
        "Please check official sources or contact your local disaster management authority. "
        f"{_CONTACTS_LINE_EN}"
    ),
    "ta": (
        "இதற்கு பதிலளிக்க எனது உள்ளூர் பேரிடர் பாதுகாப்பு வழிகாட்டுதலில் நம்பகமான தகவல் இல்லை. "
        "தயவுசெய்து அதிகாரப்பூர்வ ஆதாரங்களை பார்க்கவும் அல்லது உங்கள் உள்ளூர் பேரிடர் மேலாண்மை "
        f"அதிகாரத்தை தொடர்பு கொள்ளவும். {_CONTACTS_LINE_TA}"
    ),
    "hi": (
        "मेरे स्थानीय आपदा-सुरक्षा मार्गदर्शन से मेरे पास वह उत्तर देने के लिए विश्वसनीय जानकारी नहीं है। "
        "कृपया आधिकारिक स्रोतों की जाँच करें या अपने स्थानीय आपदा प्रबंधन प्राधिकरण से संपर्क करें। "
        "भारत में तत्काल आपातकाल में, 1070 या 1077 या 112 डायल करें।"
    ),
    "te": (
        "నా స్థానిక విపత్తు భద్రతా మార్గదర్శకత్వం నుండి ఆ ప్రశ్నకు సమాధానం ఇవ్వడానికి నాకు విశ్వసనీయ సమాచారం లేదు. "
        "దయచేసి అధికారిక వనరులను తనిఖీ చేయండి లేదా మీ స్థానిక విపత్తు నిర్వహణ అధికారిని సంప్రదించండి. "
        "భారతదేశంలో తక్షణ అత్యవసర పరిస్థితుల్లో, 1070 లేదా 1077 లేదా 112కి కాల్ చేయండి."
    ),
    "kn": (
        "ನನ್ನ ಸ್ಥಳೀಯ ವಿಪತ್ತು-ಸುರಕ್ಷತಾ ಮಾರ್ಗದರ್ಶನದಿಂದ ಆ ಪ್ರಶ್ನೆಗೆ ಉತ್ತರಿಸಲು ನನಗೆ ವಿಶ್ವಾಸಾರ್ಹ ಮಾಹಿತಿ ಇಲ್ಲ. "
        "ದಯವಿಟ್ಟು ಅಧಿಕೃತ ಮೂಲಗಳನ್ನು ಪರಿಶೀಲಿಸಿ ಅಥವಾ ನಿಮ್ಮ ಸ್ಥಳೀಯ ವಿಪತ್ತು ನಿರ್ವಹಣಾ ಪ್ರಾಧಿಕಾರವನ್ನು ಸಂಪರ್ಕಿಸಿ. "
        "ಭಾರತದಲ್ಲಿ ತುರ್ತು ಸಂದರ್ಭದಲ್ಲಿ, 1070 ಅಥವಾ 1077 ಅಥವಾ 112 ಗೆ ಕರೆ ಮಾಡಿ."
    ),
    "ml": (
        "എൻ്റെ പ്രാദേശിക ദുരന്ത നിർദേശ മാർഗ്ഗനിർദ്ദേശങ്ങൾ അനുസരിച്ച് എനിക്ക് അതിന് ഉത്തരം നൽകാൻ വിശ്വസനീയമായ "
        "വിവരങ്ങൾ ലഭ്യമല്ല. ദയവായി ഔദ്യോഗിക സ്രോതസ്സുകൾ പരിശോധിക്കുക അല്ലെങ്കിൽ നിങ്ങളുടെ പ്രാദേശിക ദുരന്ത നിവാരണ "
        "അധികാരികളെ ബന്ധപ്പെടുക. ഇന്ത്യയിൽ അടിയന്തര സാഹചര്യങ്ങളിൽ, 1070 അല്ലെങ്കിൽ 1077 അല്ലെങ്കിൽ 112 എന്ന "
        "നമ്പറിൽ വിളിക്കുക."
    ),
}


# Pool for speculative answer generation (see answer_question).
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="speculative-gen")


def _timed_generate(prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    text = ollama_client.generate(prompt)
    return text, round(time.perf_counter() - t0, 2)


@dataclass
class Answer:
    text: str
    abstained: bool
    abstention_reason: str | None
    sources: list[RetrievedChunk]
    answer_language: str = "en"
    # Kept even on abstention so evaluation can see what retrieval returned.
    search_question: str = ""
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    # Wall-clock seconds per stage: translate, retrieve, evidence_check, generate.
    timings: dict = field(default_factory=dict)
    # True when answered from the prepared instant-answer cache (no LLM call).
    cached: bool = False


def _instant_answer(retriever: HybridRetriever, question: str, language: str) -> "Answer | None":
    """Answer from the prepared cache (retrieval/answer_cache.py) if the
    question confidently matches a prepared situation; otherwise None and
    the live pipeline runs. Costs ~30 ms on a miss."""
    if not config.ANSWER_CACHE_ENABLED:
        return None
    script_lang = detect_language_from_script(question)
    # Questions about current conditions never get a prepared answer, in any
    # language (English patterns + now/today words, see live_info.py); vague
    # English questions get the clarifying reply from the live pipeline.
    if mentions_current_time(question) or (script_lang == "en" and is_too_vague(question)):
        return None
    embedder = getattr(retriever, "embedder", None)
    cache = get_answer_cache(embedder) if embedder is not None else None
    if cache is None:
        return None
    t0 = time.perf_counter()
    match = cache.match(question, retriever._embed_query(question))
    if match is None:
        return None
    lang = script_lang if language == "auto" else language
    intent = match.intent
    sources = [
        RetrievedChunk(chunk_id="", document_id=s["document_id"], disaster_category=s["disaster_category"],
                       page=s["page"], text=s["text"], bm25_score=0.0, dense_score=0.0, fused_score=0.0)
        for s in intent["sources"]
    ]
    return Answer(
        text=intent["answers"].get(lang) or intent["answers"]["en"], abstained=False, abstention_reason=None,
        sources=sources, answer_language=lang, search_question=match.matched_text,
        timings={"instant_match": round(time.perf_counter() - t0, 3), "match_score": round(match.score, 3)},
        cached=True,
    )


def answer_question(
    retriever: HybridRetriever, question: str, top_k: int = 5, language: str = "en"
) -> Answer:
    # Translate first, then retrieve in English -- see translation.py's
    # docstring for why direct non-English retrieval was unreliable. This
    # also means BM25 (impossible across scripts) now actually applies for
    # every language, not just English, so the abstention call below passes
    # query_language="en" regardless of the original question's language.
    #
    # language="auto" means "answer in whatever language the question is
    # written in" -- detect it and translate in one call, rather than
    # trusting a separately-supplied code (added so the Text tab doesn't
    # need the user to manually pick a language that should just match
    # what they typed).
    instant = _instant_answer(retriever, question, language)
    if instant is not None:
        return instant

    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if language == "auto":
        language, search_question = detect_language_and_translate(question)
    else:
        search_question = translate_to_english(question, language)
    timings["translate"] = round(time.perf_counter() - t0, 2)

    if is_too_vague(search_question):
        return Answer(
            text=CLARIFY_MESSAGE.get(language, CLARIFY_MESSAGE["en"]), abstained=True,
            abstention_reason="query_too_vague", sources=[], answer_language=language,
            search_question=search_question, timings=timings,
        )

    t0 = time.perf_counter()
    results = retriever.search(search_question, top_k=top_k)
    timings["retrieve"] = round(time.perf_counter() - t0, 2)
    assessment = assess(results, query_language=config.CORPUS_LANGUAGE)

    # Speculative generation: Ollama serves the evidence check and the answer
    # concurrently (measured 2026-10-13: 7.1 s one after the other, 4.1 s
    # together), so the answer starts generating while the check runs. If the
    # check says NO the draft is simply discarded and we abstain -- the user
    # never sees text the check rejected, so the safety behaviour is unchanged.
    prompt = build_grounded_prompt(search_question, results, answer_language=language)
    gen_future = None
    if assessment.sufficient and config.SPECULATIVE_GENERATION:
        gen_future = _POOL.submit(_timed_generate, prompt)

    if assessment.sufficient and config.EVIDENCE_CHECK_ENABLED:
        t0 = time.perf_counter()
        if not evidence_supported(search_question, results):
            assessment = EvidenceAssessment(False, "llm_evidence_check_no", results)
        timings["evidence_check"] = round(time.perf_counter() - t0, 2)

    if not assessment.sufficient:
        message = ABSTENTION_MESSAGE.get(language, ABSTENTION_MESSAGE["en"])
        return Answer(
            text=message, abstained=True, abstention_reason=assessment.reason, sources=[], answer_language=language,
            search_question=search_question, retrieved=results, timings=timings,
        )

    t0 = time.perf_counter()
    if gen_future is not None:
        response_text, timings["generate"] = gen_future.result()
        timings["generate_wait"] = round(time.perf_counter() - t0, 2)  # time spent waiting beyond the check
    else:
        response_text, timings["generate"] = _timed_generate(prompt)
    response_text = apply_live_info_guard(response_text, search_question, language=language)
    return Answer(
        text=response_text, abstained=False, abstention_reason=None, sources=results, answer_language=language,
        search_question=search_question, retrieved=results, timings=timings,
    )
