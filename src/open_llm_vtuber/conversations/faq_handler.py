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

FAQ_LIST = [
    # === บทสนทนาจาก MainConversation.md ===
    # คำตอบจะถูกอ่านผ่าน TTS (JaiTTS/EdgeTTS) เสมอ ไม่ใช้ไฟล์เสียงสำเร็จรูป
    {
        "id": "stage_1_greeting",
        "keywords": ["เริ่มพูด", "สวัสดีค่า", "เริ่มเลย", "แนะนำตัว", "เปิดรายการ", "ทักทายมาลี"],
        "answer": "สวัสดีค่า! มาลีเองนะคะ เด็กสาขา เทคโนโลยีสารสนเทศ ยินดีต้อนรับทุกคนเลยน้า วันนี้วันวิทยาศาสตร์พอดี มาลีเลยจะพาทุกคนมาวาร์ปเข้าสู่โลกดิจิทัลแห่งอนาคตกันค่ะ!",
    },
    {
        "id": "stage_2_show",
        "keywords": [
            "เตรียมอะไรมาโชว์",
            "เตรียมอะไรมา",
            "โชว์พวกเรา",
            "วันวิทยาศาสตร์แบบนี้",
            "สวัสดีครับมาลี",
        ],
        "answer": "บอกเลยว่าวันนี้มาลีจัดเต็มมากๆ ค่ะ! มาลีจะพาทุกคนมาดูฟีลแบบเรียลๆ เลยว่าระบบ AI ทำงานยังไง จะโชว์ให้เห็นกันสดๆ บนเวทีนี้เลยนะคะ ว่าสมองกลของมาลีสามารถ ฟัง คิด แล้วก็โต้ตอบกับทุกคนได้ลื่นไหลเหมือนมีชีวิตจริงๆ เลยค่ะ เตรียมว้าวกันได้เลย!",
    },
    {
        "id": "stage_3_behind_scenes",
        "keywords": [
            "ระบบเบื้องหลัง",
            "เบื้องหลังของมาลี",
            "อธิบายให้พวกเราฟัง",
            "ลื่นไหลเหมือนคนจริงๆ",
        ],
        "answer": "ระบบสมองกลของมาลีทำงานตามหลักวิทยาศาสตร์เป็นสเต็ปเลยค่ะ! เริ่มจากตอนที่ทุกคนพูดมา ระบบของมาลีจะทำ Speech Recognition คือแปลงเสียงพูดให้กลายเป็นตัวอักษรก่อน จากนั้นก็จะรีบส่งไปประมวลผลหาข้อมูลคำตอบอย่างไวในสมองกล แล้วสเต็ปสุดท้ายคือการแปลงคำตอบนั้นกลับมาเป็นเสียงน่ารักๆ ของมาลี ให้ทุกคนได้ยินกันแบบเรียลไทม์ค่ะ ทั้งหมดนี้ใช้เวลาแค่ไม่กี่มิลลิวินาทีเท่านั้นเอง ล้ำสุดๆ ไปเลยใช่ไหมล่ะ!",
    },
    {
        "id": "stage_4_applications_caution",
        "keywords": [
            "ชีวิตจริง",
            "ช่วยงานเรายังไง",
            "ข้อควรระวัง",
            " cloud พวกนี้",
            "เอามาใช้งาน",
            "เกิดขึ้นไวมาก",
            "ช่วยอะไรเราได้บ้าง",
            "ต้องระวังเป็นพิเศษ",
        ],
        "answer": "โอ้โห คำถามนี้ดีมากค่ะ! เทคโนโลยีพวกนี้ช่วยชีวิตเราได้เยอะมาก อย่าง Cloud Computing เนี่ย ฟีลมันก็เหมือนเรามี 'ตู้เซฟวิเศษบนฟ้า' ค่ะ เราเก็บข้อมูลไว้บนออนไลน์ได้เลย ไม่ต้องแบกฮาร์ดดิสก์ให้หนัก จะดึงมาใช้จากที่ไหนบนโลกก็ทำได้ ส่วน AI ก็เก่งไม่แพ้กันเลย ยิ่งในโรงพยาบาลนะคะ AI จะเข้าไปเป็นผู้ช่วยคุณหมอ ช่วยสแกนฟิล์มเอกซเรย์หาจุดผิดปกติได้อย่างแม่นยำเป๊ะๆ ทำให้รักษาผู้ป่วยได้ไวขึ้นเยอะเลยค่ะ แต่เทคโนโลยีก็มีดาบสองคมนะคะ ทริคการใช้ให้ปลอดภัยที่สุด คือ ห้ามป้อนข้อมูลส่วนตัวหรือความลับลงไปใน AI เด็ดขาด และเราต้องดับเบิลเช็กความถูกต้องของข้อมูลที่ AI ตอบมาทุกครั้งก่อนเอาไปใช้งานจริงด้วยน้า แค่นี้ก็ใช้เทคโนโลยีได้แบบเซฟๆ แล้วค่ะ!",
    },
    {
        "id": "stage_5_card_ai",
        "keywords": [
            "เห็นภาพมากขึ้น",
            "คำฮิตๆ",
            "อธิบายคำฮิต",
            "เริ่มที่ ai ก่อนเลย",
            "ตกลงมันคืออะไรครับ",
            "คำถามเรื่อง ai",
            "ตกลงแล้ว ai คืออะไร",
            "ai คืออะไร",
            "อธิบาย ai",
        ],
        "answer": "AI หรือ Artificial Intelligence คือเทคโนโลยีสมองกลที่ถูกสร้างมาให้ฉลาดเหมือนมนุษย์ค่ะ! มันไม่ได้แค่จำข้อมูลเก่งนะคะ แต่มันสามารถเรียนรู้ วิเคราะห์ แล้วก็ช่วยเราตัดสินใจแก้ปัญหาซับซ้อนได้ด้วย ฟีลเหมือนมีเพื่อนสนิทระดับอัจฉริยะคอยนั่งซัพพอร์ตเราตลอดเวลา ไม่ว่าจะช่วยทำงาน ช่วยวาดรูป หรือแม้แต่ช่วยคิดคอนเทนต์ AI ก็จัดให้ได้หมดเลยค่ะ! IoT หรือ Internet of Things ค่ะ! อธิบายง่ายๆ คือการจับเอาสิ่งของเครื่องใช้รอบตัวเรา ตั้งแต่หลอดไฟ แอร์ ไปจนถึงตู้เย็น มาเชื่อมต่อกับอินเทอร์เน็ตค่ะ ผลก็คือพวกมันจะสามารถส่งข้อมูลคุยกันเอง แล้วก็ทำงานแบบออโต้ได้เลย เช่น พอเราเดินเข้าบ้านปุ๊บ แอร์เปิด ไฟสว่าง โดยที่เราไม่ต้องกระดิกนิ้วเลยค่ะ เป็นเทคโนโลยีที่เกิดมาเพื่อสายชิลแบบพวกเราจริงๆ! Cloud Computing คือระบบจัดเก็บข้อมูลบนอินเทอร์เน็ตค่ะ! สมัยก่อนเราต้องเซฟงานลงแฟลชไดรฟ์ใช่ไหมคะ? แต่เดี๋ยวนี้เราโยนทุกอย่างขึ้น Cloud ได้เลย มันเหมือนเราเช่าพื้นที่บนฟ้าไว้เก็บไฟล์ ทำให้มือถือไม่เต็ม แถมอยากดึงงานมาทำตอนไหน หรือจะแชร์ให้เพื่อนก็ทำได้ทันที แค่ปลายนิ้วจิ้มเลยค่ะ สะดวกสุดๆ!",
    },
    {
        "id": "stage_5_card_iot",
        "keywords": [
            "คำว่า iot",
            "เทคโนโลยีแบบไหน",
            "iot",
            "ไอโอที",
        ],
        "answer": "IoT หรือ Internet of Things ค่ะ! อธิบายง่ายๆ คือการจับเอาสิ่งของเครื่องใช้รอบตัวเรา ตั้งแต่หลอดไฟ แอร์ ไปจนถึงตู้เย็น มาเชื่อมต่อกับอินเทอร์เน็ตค่ะ ผลก็คือพวกมันจะสามารถส่งข้อมูลคุยกันเอง แล้วก็ทำงานแบบออโต้ได้เลย เช่น พอเราเดินเข้าบ้านปุ๊บ แอร์เปิด ไฟสว่าง โดยที่เราไม่ต้องกระดิกนิ้วเลยค่ะ เป็นเทคโนโลยีที่เกิดมาเพื่อสายชิลแบบพวกเราจริงๆ!",
    },
    {
        "id": "stage_5_card_cloud",
        "keywords": [
            "cloud computing",
            "มันคือระบบอะไร",
            "cloud",
            "คลาวด์",
        ],
        "answer": "Cloud Computing คือระบบจัดเก็บข้อมูลบนอินเทอร์เน็ตค่ะ! สมัยก่อนเราต้องเซฟงานลงแฟลชไดรฟ์ใช่ไหมคะ? แต่เดี๋ยวนี้เราโยนทุกอย่างขึ้น Cloud ได้เลย มันเหมือนเราเช่าพื้นที่บนฟ้าไว้เก็บไฟล์ ทำให้มือถือไม่เต็ม แถมอยากดึงงานมาทำตอนไหน หรือจะแชร์ให้เพื่อนก็ทำได้ทันที แค่ปลายนิ้วจิ้มเลยค่ะ สะดวกสุดๆ!",
    },
    {
        "id": "stage_6_closing_1",
        "keywords": [
            "ฉลาดกว่าที่คิด",
            "มีประโยชน์และฉลาด",
            "ฉลาดกว่าที่คิดเยอะเลยครับ",
        ],
        "answer": "ใช่แล้วล่ะค่ะ! แต่ถึง AI จะเก่งแค่ไหน เทคโนโลยีก็ไม่ได้ถูกสร้างมาเพื่อแย่งงานคนนะคะ แต่มันคือ 'เครื่องมือ' ที่มาช่วยซัพพอร์ตให้เราทำงานได้ปังขึ้นต่างหากค่ะ ในฐานะเด็กสาขา IT มาลีเชื่อว่าถ้าเราใช้เทคโนโลยีอย่างสร้างสรรค์ โลกอนาคตของเราจะต้องล้ำหน้าและน่าอยู่มากๆ แน่นอนค่ะ!",
    },
    {
        "id": "stage_6_closing_2",
        "keywords": [
            "digital technology",
            "human creativity",
            "ความคิดสร้างสรรค์",
            "คืออนาคตของเราครับ",
        ],
        "answer": "ใครที่อยากลองเล่นเทคโนโลยีเจ๋งๆด้วยตัวเอง แวะไปเจอกันที่บูธกิจกรรมด้านหน้าได้เลยนะคะ! วันนี้วัน วิทยาศาสตร์ ขอให้ทุกคนสนุกกับโลกดิจิทัลน้า มาลีต้องขอตัวไปก่อนแล้ว บ๊ายบายค่า!",
    },
    {
        "id": "stage_6_closing_3",
        "keywords": ["ขอเสียงปรบมือ", "ปรบมือให้น้องมาลี", "ขอบคุณทุกคนมากครับ", "ปรบมือ"],
        "answer": "ขอบคุณมากค่ะ",
    },
]


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


def calc_similarity(kw: str, text: str) -> float:
    """Calculate similarity percentage (0.0 to 100.0) between keyword and user input text.

    Rules (strict, accuracy over recall):
    1. Ambiguous/generic keywords never trigger on their own.
    2. An exact substring match scores 100%.
    3. Otherwise the whole sentence must be roughly the same length as the
       keyword and similar to it. A long, differently-worded sentence can never
       match — sliding-window partial matches are NOT allowed.
    """
    kw_clean = kw.lower().strip()
    text_clean = text.lower().strip()

    if not kw_clean or not text_clean:
        return 0.0

    if kw_clean in _WEAK_KEYWORDS:
        return 0.0

    if _find_substring(kw_clean, text_clean):
        return 100.0

    # Phrase-level match only: the input must be roughly the same length as the
    # keyword (allows slight rewording) — never a partial window inside a longer
    # sentence that merely resembles the keyword.
    if len(text_clean) <= len(kw_clean) * 2:
        return difflib.SequenceMatcher(None, kw_clean, text_clean).ratio() * 100.0

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

    return None
