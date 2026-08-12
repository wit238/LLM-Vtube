import re
import difflib
import time
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger

BASE_DIR = Path(__file__).parent.parent.parent.parent
AUDIO_DIR = BASE_DIR / "assets" / "faq_audio"
GREETING_AUDIO_PATH = str(AUDIO_DIR / "greeting.mp3")

# A pre-rendered FAQ answer is only played *once* per window. If a question that
# matches the same FAQ is asked again within the window, the conversation falls
# through to the real LLM instead of replaying the canned answer.
FAQ_REPEAT_COOLDOWN_SECONDS = 600.0
_recent_faq_triggers: Dict[str, float] = {}


def reset_faq_cooldown() -> None:
    """Clear the repeat-answer cooldown so every FAQ can trigger again.

    Called when a new chat session starts (or the handler is re-enabled), so
    the "+" button makes scripted answers available immediately even if the
    same question was asked within the cooldown window.
    """
    _recent_faq_triggers.clear()

_ASCII_RE = re.compile(r"[A-Za-z0-9]")

# Generic / ambiguous keywords. They appear as substrings inside *other* intents
# (e.g. "ที่ไหน" in "สมัครที่ไหน", "กี่บาท" in "ซื้อคอมกี่บาท") so they must
# NEVER trigger a pre-rendered FAQ on their own. The LLM answers those instead.
_WEAK_KEYWORDS = {
    "ที่ไหน",
    "อยู่ที่ไหน",
    "ตึกไหน",
    "ตั้งอยู่ที่",
    "กี่บาท",
    "กี่ปี",
    "แพงไหม",
    "เท่าไหร่",
    "เท่าไร",
    "ต่างกันยังไง",
    "ต่างกันอย่างไร",
    "ทำงานยังไง",
    "ทำงานอะไร",
    "มันคืออะไร",
    "สรุปแบบเข้าใจง่ายๆ",
    "เข้าใจง่ายๆ หน่อยครับ",
    "เห็นภาพมากขึ้น",
    "เห็นภาพหน่อยครับ",
    "เรียนอะไรบ้าง",
    "ยากไหม",
    "เรียนยาก",
}

def _load_faq_list() -> list[dict]:
    """Build the FAQ list from knowledge_md/MainConversation.md.

    Every "**พิธีกร:**" question mapped to the following "**มาลี:**" answer
    becomes one FAQ entry. Keywords = the full question plus its clauses
    split on " หรือ / และ / แล้ว " (exact script lines score 100%). If the
    file is missing or unreadable, the FAQ handler is disabled (no matches).
    """
    md_path = BASE_DIR / "knowledge_md" / "MainConversation.md"
    if not md_path.is_file():
        logger.warning(f"faq_handler: {md_path} not found - FAQ disabled")
        return []

    try:
        lines = md_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        logger.warning(f"faq_handler: cannot read {md_path}: {e} - FAQ disabled")
        return []

    pairs: list[list[str]] = []
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current, buf
        text = " ".join(p.strip() for p in buf).strip()
        while True:
            before = text
            text = text.strip("“”\"' ")
            if text.endswith("**"):
                text = text[:-2].strip()
            if text == before:
                break
        if current == "host" and text:
            pairs.append([text, ""])
        elif current == "malinee" and pairs:
            pairs[-1][1] = text
        current = None
        buf = []

    for line in lines:
        s = line.strip()
        if s.startswith("**พิธีกร") or s.startswith("**มาลี"):
            flush()
            current = "host" if s.startswith("**พิธีกร") else "malinee"
            rest = s.split(":", 1)[-1].strip()
            if rest:
                buf.append(rest)
        elif current and (s.startswith("---") or s.startswith("###") or s.startswith("**")):
            flush()
        elif current:
            buf.append(s)
    flush()

    faqs = []
    for i, (question, answer) in enumerate(pairs):
        if not question or not answer:
            continue
        keywords = [question]
        for sep in (" หรือ ", " และ ", " แล้ว "):
            for part in question.split(sep)[1:]:
                part = part.strip(" ,.?!?")
                if len(part) >= 6:
                    keywords.append(part)
        seen, unique = set(), []
        for kw in keywords:
            key = kw.lower()
            if key not in seen:
                seen.add(key)
                unique.append(kw)
        faqs.append(
            {
                "id": f"faq_{i + 1:02d}",
                "keywords": unique,
                "answer": answer,
            }
        )
    logger.info(f"faq_handler: loaded {len(faqs)} FAQ(s) from {md_path.name}")
    return faqs


# Built from knowledge_md/MainConversation.md at import time. Answers are
# always spoken through TTS (never pre-rendered audio files).
FAQ_LIST = _load_faq_list()

# ---- Semantic matching (conference-style) ----
# When the keyword scorer misses, embed the user's question and compare it
# against the scripted questions (same embedding model as the RAG knowledge
# base - BAAI/bge-m3, cached as a singleton, so warm embeds take ~0.1s).
# Calibrated against 10 real probes: true paraphrases score 0.83-0.90 while
# unrelated Thai sentences max out at ~0.73 (bge-m3 inflates short-sentence
# similarity), so 0.80 + a 0.05 margin over the runner-up cleanly separates
# conference-style follow-ups from chit-chat.
_SEMANTIC_THRESHOLD = 0.80  # bge-m3 cosine similarity
_SEMANTIC_MARGIN = 0.05  # top score must beat runner-up by at least this
_semantic_vectors = None  # (N, D) float32, L2-normalized
_semantic_ready = False


def _load_semantic_vectors() -> None:
    """Embed every FAQ's primary question once (lazy, cached)."""
    global _semantic_vectors, _semantic_ready
    if _semantic_ready:
        return
    _semantic_ready = True
    if not FAQ_LIST:
        return
    try:
        from ..knowledge.embedder import embed_texts

        questions = [faq["keywords"][0] for faq in FAQ_LIST]
        _semantic_vectors = embed_texts(questions)
        logger.info(
            f"faq_handler: semantic vectors ready ({len(questions)} questions)"
        )
    except Exception as e:
        logger.warning(
            f"faq_handler: semantic matching unavailable ({e}) - "
            "keyword matching only"
        )


def _semantic_match(text: str):
    """Return (faq, cosine_score) for the closest scripted question, or None."""
    _load_semantic_vectors()
    if _semantic_vectors is None:
        return None
    try:
        from ..knowledge.embedder import embed_texts

        q = embed_texts([text])[0]
        scores = _semantic_vectors @ q
        order = scores.argsort()[::-1]
        idx = int(order[0])
        score = float(scores[idx])
        second = float(scores[order[1]]) if len(order) > 1 else 0.0
        if score >= _SEMANTIC_THRESHOLD and (score - second) >= _SEMANTIC_MARGIN:
            return FAQ_LIST[idx], score
    except Exception as e:
        logger.warning(f"faq_handler: semantic match failed ({e})")
    return None


def _find_substring(kw: str, text: str) -> bool:
    """Find keyword in text.

    ASCII keywords (e.g. "ai", "iot", "cloud") are matched on word boundaries
    only, so "it"/"ai" never match inside "wait"/"again". Thai keywords have no
    word boundaries (words are written without spaces), so they match anywhere
    in the text.
    """
    if not kw or not text:
        return False
    first_ascii = bool(_ASCII_RE.match(kw[0]))
    last_ascii = bool(_ASCII_RE.match(kw[-1]))
    if first_ascii or last_ascii:
        escaped = re.escape(kw)
        if first_ascii and last_ascii:
            pattern = rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
        elif first_ascii:
            pattern = rf"(?<![A-Za-z0-9]){escaped}"
        else:
            pattern = rf"{escaped}(?![A-Za-z0-9])"
        return re.search(pattern, text) is not None
    return kw in text


_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_NON_THAI_RE = re.compile(r"[^\u0E00-\u0E7F]+")


def _thai_span(s: str) -> str:
    """Keep only Thai characters (Thai words are written without spaces)."""
    return _NON_THAI_RE.sub("", s.lower())


def _split_keyword_units(kw: str) -> list:
    """Split a keyword into semantic units: ASCII words + Thai spans.

    "ai คืออะไร" -> ["ai", "คืออะไร"]; "cloud computing" -> ["cloud", "computing"]
    """
    units = []
    for part in _ASCII_TOKEN_RE.split(kw):
        part = _thai_span(part)
        if part:
            units.append(part)
    units.extend(_ASCII_TOKEN_RE.findall(kw.lower()))
    return units


def _unit_in_text(unit: str, text: str, thai_text: str) -> bool:
    """Check whether one semantic unit appears in the input text."""
    if not unit:
        return False
    if _ASCII_TOKEN_RE.match(unit):
        return _find_substring(unit, text)
    return unit in thai_text


def calc_similarity(kw: str, text: str) -> float:
    """Calculate similarity percentage (0.0 to 100.0) between keyword and user input text.

    Rules (word-level, accuracy over recall):
    1. Ambiguous/generic keywords never trigger on their own.
    2. An exact substring match scores 100%.
    3. Word coverage: split the keyword into semantic units (ASCII words +
       Thai spans) and score the fraction of units found in the input.
    4. Otherwise the whole sentence must be roughly the same length as the
       keyword and similar to it (fallback for short reworded inputs).
    """
    kw_clean = kw.lower().strip()
    text_clean = text.lower().strip()

    if not kw_clean or not text_clean:
        return 0.0

    if kw_clean in _WEAK_KEYWORDS:
        return 0.0

    if _find_substring(kw_clean, text_clean):
        return 100.0

    # Word-level coverage: how many of the keyword's units appear in the input
    units = _split_keyword_units(kw_clean)
    if len(units) == 1 and len(kw_clean) <= 2:
        # Single 1-2 char units (e.g. "ai") are too ambiguous on their own
        return 0.0
    if units:
        thai_text = _thai_span(text_clean)
        matched = sum(1 for u in units if _unit_in_text(u, text_clean, thai_text))
        coverage = matched / len(units)
        if coverage >= 0.66:
            return coverage * 100.0

    # Phrase-level match only: the input must be roughly the same length as the
    # keyword and *nearly identical* to it (allows slight rewording). The bar is
    # high (>= 0.75) because short Thai phrases share many characters by chance
    # ("เรียนอะไรบ้าง" vs "เตรียมอะไรมา" ~0.72) and must not trigger.
    if len(text_clean) <= len(kw_clean) * 2:
        ratio = difflib.SequenceMatcher(None, kw_clean, text_clean).ratio()
        if ratio >= 0.75:
            return ratio * 100.0

    return 0.0


def match_faq(
    user_input: str, similarity_threshold_percent: float = 60.0
) -> Optional[Dict[str, Any]]:
    """Check if user input matches any FAQ keywords based on exact substring or similarity % threshold."""
    if not user_input or not isinstance(user_input, str):
        return None

    clean_text = user_input.lower().strip()
    best_match = None
    highest_score = 0.0
    best_keyword = ""

    for faq in FAQ_LIST:
        for kw in faq.get("keywords", []):
            score = calc_similarity(kw, clean_text)
            if score > highest_score:
                highest_score = score
                best_match = faq
                best_keyword = kw
            elif score == highest_score and score > 0.0 and len(kw) > len(best_keyword):
                # Same score -> prefer the more specific (longer) keyword so a
                # vague term of another FAQ doesn't win over a precise one.
                best_match = faq
                best_keyword = kw

    if best_match and highest_score >= similarity_threshold_percent:
        faq_id = best_match["id"]
        now = time.monotonic()
        last = _recent_faq_triggers.get(faq_id)
        if last is not None and (now - last) < FAQ_REPEAT_COOLDOWN_SECONDS:
            logger.info(
                f"♻️ FAQ {faq_id} repeated within {FAQ_REPEAT_COOLDOWN_SECONDS:.0f}s "
                "cooldown — deferring to the real LLM"
            )
            return None

        _recent_faq_triggers[faq_id] = now
        logger.info(
            f"🎯 FAQ Match found: {faq_id} (Score: {highest_score:.1f}% >= Threshold {similarity_threshold_percent:.1f}%)"
        )
        return best_match

    # Conference-style fallback: no keyword hit, but the user's question is
    # semantically similar to a scripted one -> answer from MainConversation.md.
    sem = _semantic_match(clean_text)
    if sem is not None:
        best_match, cos = sem
        faq_id = best_match["id"]
        now = time.monotonic()
        last = _recent_faq_triggers.get(faq_id)
        if last is not None and (now - last) < FAQ_REPEAT_COOLDOWN_SECONDS:
            logger.info(
                f"♻️ FAQ {faq_id} repeated within {FAQ_REPEAT_COOLDOWN_SECONDS:.0f}s "
                "cooldown — deferring to the real LLM"
            )
            return None
        _recent_faq_triggers[faq_id] = now
        logger.info(
            f"🎯 FAQ Match found (semantic): {faq_id} "
            f"(cosine {cos:.2f} >= {_SEMANTIC_THRESHOLD:.2f})"
        )
        return best_match

    return None


# Pre-warm the semantic vectors at import time so the first conference-style
# question doesn't pay the ~10s embedding cost (the model itself is already
# loaded by the RAG knowledge base, which shares the same singleton).
_load_semantic_vectors()
