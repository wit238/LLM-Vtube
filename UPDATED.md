# UPDATED.md — สรุปทุกสิ่งที่เพิ่ม/แก้ไขในโปรเจกต์ (ไล่ตามช่วงเวลา)

เอกสารนี้สรุปงานทั้งหมดที่ทำใน **Open-LLM-VTuber** โฟลเดอร์นี้ ตั้งแต่
เริ่มอินทิเกรต JaiTTS จนถึงระบบ RAG — เรียงตามลำดับจริงที่ทำ

---

## ช่วงที่ 1: อินทิเกรต JaiTTS (TTS / F5-TTS voice clone ภาษาไทย) เข้า TTS Pipeline

### 1.1 สร้าง TTS Engine ใหม่ `jaitts_tts`
- **ไฟล์ใหม่**: `src/open_llm_vtuber/tts/jaitts_tts.py`
- HTTP client เรียก JaiTTS server ในเครื่อง (`http://127.0.0.1:8021/synthesize`)
- รับพารามิเตอร์: `api_url`, `ref_audio_path` (เสียงอ้างอิงสำหรับ voice cloning), `ref_text` (บทของเสียงอ้างอิง), `speed`, `seed`, `timeout`
- POST text + reference voice → บันทึก WAV ลง `cache/`
- ลงทะเบียนใน `tts_factory.py` (case `jaitts_tts`) และ `TTSConfig` → เพิ่ม `"jaitts_tts"` ใน Literal

### 1.2 Auto-start JaiTTS server ตอนเปิด VTuber server
- **แก้**: `run_server.py`
- เพิ่ม `ensure_jaitts_server(config)` — ถ้า config เลือก `jaitts_tts` และ `auto_start: true`:
  - เช็ค `/health` ก่อน ถ้า healthy แล้ว → ข้าม
  - spawn ตัว `server_local.py` ของโปรเจกต์ `jaitts_tools` (ใช้ **CUDA venv แยกของตัวเอง**) เป็น subprocess
  - `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW` (Windows), log ไป `logs/jaitts_server.log`
  - **กันการชนกันของ HF cache**: ลบ `HF_HOME`/`HF_HUB_CACHE`/`MODELSCOPE_CACHE` ออกจาก child env เพราะ JaiTTS ใช้ cache เริ่มต้นของตัวเอง
  - คอย health check จนพร้อม (default timeout 300s) → `stop_jaitts_server()` ลงทะเบียน `atexit`
- โฟลเดอร์ server หาได้จาก `server_dir` หรือเดาจาก `ref_audio_path` (เช่น `jaitts_tools`)

### 1.3 แก้ห่วง JaiTTS server ค้าง/ล็อกไฟล์ log
- ปัญหา: process cmd เก่า (PID 17424) ค้างที่หน้าจอ *"Terminate batch job (Y/N)?"* แล้วล็อกไฟล์ `jaitts_server.log` ตลอด
- แก้ `_health_ok()` ให้ `except BaseException` (เดิม catch `Exception` ตัว `KeyboardInterrupt` ซึ่งเป็น `BaseException` ทำให้ค้างได้) — ผลลัพธ์: health check ไม่ค้าง ไม่ตายตอน interrupt

### 1.4 Config JaiTTS
- **ไฟล์**: `config_manager/tts.py` → คลาส `JaiTTSTTSConfig` (i18n description en/zh)
- field: `api_url`, `ref_audio_path` (required), `ref_text`, `speed` (default 1.0), `seed` (-1), `timeout` (300s), `auto_start` (true), `server_dir`, `server_startup_timeout` (300s)
- `conf.yaml` + templates เพิ่มบล็อก `jaitts_tts:`
- **ค่าจริงในเครื่อง**: `ref_audio_path: jaitts_tools\ref.wav` + ref_text ภาษาไทย (เสียง voice clone ของ "มาลี"), `server_dir: ...\jaitts_tools`, `auto_start: true`

---

## ช่วงที่ 2: ASR ภาษาไทย (Typhoon ASR)

### 2.1 สร้าง ASR engine ใหม่ `typhoon_asr`
- **ไฟล์ใหม่**: `src/open_llm_vtuber/asr/typhoon_asr.py`
- ใช้ package `typhoon-asr` (โมเดล `scb10x/typhoon-asr-realtime`)
- เขียน float array → temp WAV → `transcribe()` → ดีง text ออกจาก result แบบ recursive (รองรับ str/dict/object/list)
- **แก้**: `asr_factory.py` → `elif system_name == "typhoon_asr"`
- **แก้**: `config_manager/asr.py` → คลาส `TyphoonASRConfig` + เพิ่ม `"typhoon_asr"` ใน Literal `ASRConfig`
- **แก้**: `pyproject.toml` → `"typhoon-asr>=0.1.1"` (ติดตั้งสำเร็จ)
- **ค่าจริงในเครื่อง**: `device: cuda` (ใช้ GPU), `model_name: scb10x/typhoon-asr-realtime`

---

## ช่วงที่ 3: เลือก Agent + LLM + ตั้งบุคลิก

### 3.1 Agent & LLM
- `conversation_agent_choice: basic_memory_agent`
- LLM ใช้ `openai_compatible_llm` → base_url `https://opencode.ai/zen/v1`, model `deepseek-v4-flash-free`, key จาก env `${OPENCODE_API_KEY}`
- ที่เหลือใน `llm_configs` (claude, ollama, gemini, ฯลฯ) เป็นตัวเลือกเหลือไว้เปลี่ยนได้
- VAD: `vad_model: null` (ปิด)

### 3.2 FAQ ไทย
- `conversations/faq_handler.py` มีรายการ FAQ ไทย hardcoded (`FAQ_LIST`) เช่น ค่าเทอม (`faq_tuition`), การสมัคร (`faq_apply`)
- จับคู่ด้วย difflib keyword, threshold default 60% → ตอบสคริปต์+เสียงสำเร็จรูปทันที

---

## ช่วงที่ 4: Auto-trim เสียง WAV ที่ออกจาก JaiTTS

### 4.1 สร้าง `utils/audio_trim.py`
- **Port จาก** `jaitts_tools\trim_leading.py`
- วิธี: `faster-whisper` `word_timestamps=True` → transcript ไทย → จับคู่กับข้อความต้นฉบับด้วย `difflib.SequenceMatcher` → ตัดก่อนคำแรกที่ match และหลังคำสุดท้ายที่ match
- Fallback: ตัด tail noise ด้วยพลังงาน (`trailing_energy_start`) เมื่อหา end-word ไม่เจอ
- **Safety**:
  - `MIN_MATCH_RATIO=0.35` — transcript ไม่ตรงข้อความ → SKIP ไม่ตัด (กันตัดคำพูดจริง)
  - ตัดเฉพาะเมื่อส่วนเกิน ≥ `min_cut` (0.30s), เหลือหน้า `HEADROOM=0.10s`, ท้าย `TAIL_MARGIN=0.15s`
- `_get_whisper_model()` lazy + cache singleton (มี lock), กัน error สวยๆ ถ้าไม่ได้ติดตั้ง faster-whisper

### 4.2 wire เข้า `jaitts_tts`
- `generate_audio()` → หลังได้ WAV → `_trim_wav(out, text)` (ข้ามถ้าคลิปสั้นกว่า 0.5s; failure = คง WAV เดิม, ไม่ทำให้ TTS พัง)
- **Config เพิ่ม** ใน `JaiTTSTTSConfig`: `trim_audio`, `trim_model` (small), `trim_device` (cpu), `trim_compute_type` (int8), `trim_download_root` (models/whisper), `trim_headroom`, `trim_tail_margin`, `trim_min_cut`
- `tts_factory.py` ส่งค่าเข้า engine
- **ผลจริง**: ตัดหัวเงียบได้ 0.32–0.38s/clip; ครั้งแรกหลัง cold ~116s (โหลดโมเดล), warm ~7s

### 4.3 Dependency
- `pyproject.toml` → `faster-whisper>=1.1.0` (ติดตั้ง 1.2.1 + ctranslate2 4.8.1)

---

## ช่วงที่ 5: File Knowledge Base (RAG) จากไฟล์ Markdown

### 5.1 ภาพรวม
- วาง `.md` ในโฟลเดอร์ (`knowledge_md/`) → index อัตโนมัติตอน start → เวลาคุย retrieve chunk ที่ตรงกับคำถาม ใส่ context ให้ LLM ตอบจากเอกสารโดยไม่เพ้อเจ้อ

### 5.2 ไฟล์ใหม่ `src/open_llm_vtuber/knowledge/`
| ไฟล์ | หน้าที่ |
|---|---|
| `chunker.py` | ตัด md เป็น chunk ตาม heading (ซ้อนเป็น path), `chunk_size=1000`/`overlap=150`, ตัด code fence; อ่านไฟล์ซ้ำได้ (subfolder) |
| `embedder.py` | `BAAI/bge-m3` ผ่าน transformers เดิม, mean-pool + L2 normalize, lazy singleton + lock |
| `knowledge_base.py` | `build_index()` hash-based re-index + cache ถาวร (`cache/knowledge/index.json` + `vectors.npy`), warm-up โมเดลตอน start; `retrieve()` cosine (numpy `@`), `top_k`, `min_score` |
| `base.py` | ตัวเข้าถึง singleton ระดับ module (`set/get_knowledge_base`) |

### 5.3 Wire เข้าระบบ
- **`run_server.py:288-302`**: สร้าง `KnowledgeBase` ตอน startup หลัง `ensure_jaitts_server`
- **`single_conversation.py:138-160`**: retrieve → `batch_input.metadata["knowledge_context"]` พร้อมหัวไทย "ข้อมูลอ้างอิง (จงใช้ข้อมูลนี้เป็นความรู้ของตัวเอง...)" — **ห้ามเปิดเผยแหล่งที่มา/ห้ามตอบว่าไม่รู้**; ถ้า folder เล็ก (≤12 chunks) **ใส่ความรู้ทั้งหมด** ไม่ retrieve บางส่วน
- **`basic_memory_agent.py:238-243`**: `_to_messages` แทรก knowledge context เป็นข้อความ user เพิ่ม (อยู่แค่ metadata → **ไม่ถูกบันทึกเป็น memory**)
- **`service_context.py:474-480`**: เมื่อ `knowledge_config.enabled` → เพิ่มคำสั่งไทยใน system prompt: ใช้ "ข้อมูลอ้างอิง" เป็นความรู้ของตัวเอง ตอบเป็นธรรมชาติ **ห้ามพูดถึงเอกสาร/ไฟล์/คลังข้อมูล และห้ามตอบว่า "ไม่รู้"**

### 5.4 Config
- **ใหม่**: `config_manager/knowledge.py` → `KnowledgeConfig` (enabled, folder_path, embedding_model=BAAI/bge-m3, embedding_device=cpu, chunk_size/overlap, top_k=8, min_score=0.2, cache_dir, extension, include_sources=false)
- `CharacterConfig` + `knowledge_config: Optional[KnowledgeConfig]`, ให้ `ConfigDict(extra="allow")`
- export ใน `config_manager/__init__.py`
- `conf.yaml` + templates เพิ่มบล็อก `knowledge_config:`
- ตัวอย่าง 2 ไฟล์: `knowledge_md/วันสำคัญ.md`, `knowledge_md/นครปฐม-ไอที.md`

### 5.5 ผลตรวจวัด
- Retrieval ดี: ค่าเทอม→0.82, รับสมัคร→0.75, สถาปนา→0.75; คำถามนอกคลัง "ขายข้าวราดแกง" → 0.57 (filter โดย min_score)
- Latency: cold embed-model load ~11-12s (กันด้วย warm-up ตอน start), warm embed ~0.1s, retrieve ~0.1s
- E2E: คำถามที่ FAQ ไม่จับ ("วันปฐมนิเทศ...") → ผ่าน `chat_with_memory` → ถึง LLM → JaiTTS ตอบ; คำถามที่ FAQ จับ ("ค่าเทอม/สมัคร") → ตอบ FAQ script ตาม design

---

## ช่วงที่ 6: ปรับ Persona Prompt ให้เป็นหมวดหมู่ + บังคับ SFW

- แก้ `persona_prompt` ใน `conf.yaml` → โครงสร้างเป็นหมวด:
  `ตัวตน → บุคลิกภาพ → ลักษณะการพูด → พฤติกรรม → สิ่งที่ชอบ/ไม่ชอบ → พื้นหลัง → ความรู้/FAQ → กฎ SFW 100%`
- บุคลิก: "มาลี" นิสิตไอที มรน. น่ารัก สดใส ขยันเรียน
- เพิ่ม **กฎ SFW 5 ข้อ**: ห้าม NSFW/อนาจาร/ลวนลามโดยเด็ดขาด → ปฏิเสธสุภาพแล้วเปลี่ยนเรื่อง; ตอบตามข้อมูลจากเอกสาร/FAQ เท่านั้น; ตรวจการสะกดไทยให้ถูก; เป็นต้น
- ความยาว 4181 ตัวอักษร
- *ต้อง restart server ถึงมีผล*

---

## ช่วงที่ 7: งานแก้/ปรับปรุงระบบเดิมระหว่างทาง

- **Health check ฝั่ง JaiTTS** ไม่ค้างเมื่อมี signal ขัดจังหวะ (catch `BaseException`)
- **Encoding**: แก้ปัญหาอักขระไทยเพี้ยนเวลาอ่าน/เขียน YAML ภาษาไทย (ทั้ง config validation และ log)
- **Upgrade system**: อัปเดต config template (EN/ZH) ให้ตรงกับ field ใหม่ทุกครั้ง (`conf.default.yaml`, `conf.ZH.default.yaml`)
- **VAD ปิด** ใช้โฟลว์ speech detection ปกติของ server

---

## ช่วงที่ 8: รองรับ Railway Deployment

- **ใหม่**: `config_templates/conf.railway.yaml` — template ที่ auto-selected บน Railway
  - คัดลอกจาก `conf.yaml` จริง (agent/LLM/ASR/TTS/FAQ/RAG เดิมครบ) แต่แทนที่ path เฉพาะเครื่องและ key จริงด้วย env var:
    - `OPENCODE_API_KEY` (LLM), `JAITTS_REF_AUDIO`, `JAITTS_SERVER_DIR`
    - optional: `FISH_API_KEY`, `ELEVENLABS_API_KEY`
  - ใช้ `read_yaml` resolve `${ENV}` เดิมของโปรเจกต์ (ไม่ต้องแก้ engine)
  - ไม่มี secret/path เครื่องหลุด (ตรวจแล้ว)
- **แก้**: `scripts/start-app.sh` — ลำดับเลือก conf.yaml:
  1. `/app/conf/conf.yaml` (volume/user) → 2. repo `conf.yaml` → 3. **Railway auto-detect** (`RAILWAY_ENVIRONMENT`/`PROJECT_ID`/`SERVICE_ID` → ใช้ `conf.railway.yaml`) → 4. default template
- **Dockerfile**: มีอยู่แล้ว (python:3.10-slim + uv + start-app.sh, EXPOSE 12393) — `run_server.py` อ่าน `PORT` env อยู่แล้ว (Railway inject `PORT` → ใช้ได้ทันที)
- `.dockerignore` เก็บ `config_templates/` + `knowledge_md/` ไว้ใน image อยู่แล้ว
- *หมายเหตุ*: JaiTTS/Typhoon ต้องการ GPU + local server — บน Railway ที่ไม่มี GPU ต้องตั้ง `auto_start: false` + `api_url` ชี้ server ภายนอก (มี note ใน template)

---

## ไฟล์ใหม่ทั้งหมด (ไม่รวมของเดิม)
- `src/open_llm_vtuber/tts/jaitts_tts.py`
- `src/open_llm_vtuber/asr/typhoon_asr.py`
- `src/open_llm_vtuber/utils/audio_trim.py`
- `src/open_llm_vtuber/knowledge/` (5 ไฟล์: `chunker.py`, `embedder.py`, `knowledge_base.py`, `base.py`, `__init__.py`)
- `src/open_llm_vtuber/config_manager/knowledge.py`
- `knowledge_md/วันสำคัญ.md`, `knowledge_md/นครปฐม-ไอที.md`

## ไฟล์ที่แก้
- `run_server.py` — auto-start JaiTTS + build KB + `_health_ok` fix
- `src/open_llm_vtuber/tts/tts_factory.py`, `tts/jaitts_tts.py` — engine + trim wire
- `src/open_llm_vtuber/asr/asr_factory.py` — typhoon
- `src/open_llm_vtuber/config_manager/` — `tts.py`, `asr.py`, `character.py`, `__init__.py`
- `src/open_llm_vtuber/conversations/single_conversation.py` — RAG retrieval
- `src/open_llm_vtuber/agent/agents/basic_memory_agent.py` — knowledge context ใน `_to_messages`
- `src/open_llm_vtuber/service_context.py` — คำสั่งไทย system prompt
- `conf.yaml`, `config_templates/conf.default.yaml`, `conf.ZH.default.yaml`

## Dependency เพิ่ม
- `typhoon-asr>=0.1.1`
- `faster-whisper>=1.1.0`

---

## หมายเหตุ
- Server จึงตอบเสียงไทยได้ครบ: LLM (deepseek) → ข้อความไทย → JaiTTS (voice clone มาลี) + trim → พูด
- ASR ใช้ Typhoon realtime บน cuda, VAD ปิด
- คำถามที่ตรง FAQ จะถูกตอบจาก FAQ script ก่อนถึง RAG — ถ้าอยากให้เอกสารชนะ FAQ ต้องปรับลำดับการเช็ค
- รายละเอียดแยกแต่ละระบบอยู่ใน `conf.yaml` (comment) และคลาส config ใน `config_manager/`

---

## ช่วงที่ 8: เร่งความเร็ว JaiTTS (nfe_step) + JaiTTS เป็น local เต็มรูปแบบ

### 8.1 JaiTTS เป็น local (ไม่ใช้ Modal)
- `conf.yaml`: `api_url: 'http://127.0.0.1:8021/synthesize'`, `auto_start: true`, `server_dir: jaitts_tools`
- `run_server.py ensure_jaitts_server` spawn `server_local.py` (CUDA venv ของ jaitts_tools) → health check → พร้อม ~70s; ปิด server → ปิดตาม
- Railway เข้า localhost ไม่ได้ → edge_tts fallback อัตโนมัติ (แก้ comment ใน railway template)

### 8.2 nfe_step (flow-matching ODE steps) — ตัวหลักของการเร่ง
- Benchmark จริงบน RTX 2050 (เสียง 4.6s): `32 → 9.4s`, `24 → 6.2s` (คุณภาพเท่าเดิม), `16 → 4.0s` (~2.4x, มี babble หัวเสียงเป็นครั้งคราว)
- **Default เปลี่ยน 32 → 24** ใน `server_local.py` + `run_local.py` (constant `NFE_STEP`)
- `/synthesize` รับ form field `nfe_step` เพิ่ม — client `jaitts_tts.py` ส่งค่าได้จาก config
- Config ใหม่ `nfe_step: 24` ใน `JaiTTSTTSConfig` + conf.yaml + templates