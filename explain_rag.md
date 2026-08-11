# 📖 คู่มือและขบวนการทำ RAG จากไฟล์ Markdown (Markdown to RAG Pipeline)

การทำ **RAG (Retrieval-Augmented Generation)** จากไฟล์ **Markdown (.md)** คือกระบวนการดึงข้อความจากเอกสารโครงสร้าง Markdown มาย่อย แปลงเป็นข้อมูลเวกเตอร์ความหมาย (Embedding) และค้นหาข้อมูลที่เกี่ยวข้องที่สุดเพื่อนำไปเสริมใน Prompt ให้ **LLM (Large Language Model)** ตอบคำถามได้อย่างแม่นยำและอิงตามเอกสารจริง

---

## 📌 1. สรุปภาพรวมใน 1 นาที (TL;DR)

```
[ เอกสาร Markdown (.md) ]
       │
       ▼
[ 1. Document Loading & Preprocessing ] ── (ลบส่วนเกิน เช่น Code Fences)
       │
       ▼
[ 2. Header-Based Chunking ] ──────────── (แบ่งชิ้นส่วนข้อความตามหัวข้อ # ## ###)
       │
       ▼
[ 3. Vector Embedding ] ────────────────── (แปลงข้อความ Chunk เป็น Dense Vector)
       │
       ▼
[ 4. Vector Storage & Cache ] ─────────── (บันทึก Vector + Metadata ลงไฟล์/VectorDB)
       │
       ▼
[ 5. Retrieval & Context Injection ] ──── (ดึง Top-K Chunks ส่งใส่ LLM Prompt)
```

---

## 🔄 2. ขบวนการและขั้นตอนการทำงานแบบละเอียด (5 Steps)

### Step 1: Document Loading & Preprocessing (อ่านและทำความสะอาด)
* อ่านไฟล์ `.md` ทั้งหมดในโฟลเดอร์เป้าหมายแบบกระจายซ้อนกัน (Recursive Scanning)
* ลบสัญลักษณ์หรือขยะที่ไม่จำเป็น เช่น Code Fences (` ``` ` หรือ ` ~~~ `) เพื่อไม่ให้รบกวนความหมายเวกเตอร์

### Step 2: Markdown-Aware Chunking (การตัดแบ่งข้อความตามโครงสร้าง)
* **ความท้าทาย:** หากตัดแบ่งตามจำนวนตัวอักษรธรรมดา ข้อความอาจถูกตัดขาดกลางประโยคหรือหลุดบริบทหัวข้อ
* **แนวทางแก้ไข:** สแกนหา Header (`#`, `##`, `###`) เพื่อสร้าง **Heading Hierarchy (ลำดับหัวข้อ)** แล้วแปะ Breadcrumb กำกับไว้ในทุก Chunk เช่น:
  ```text
  [document.md > หัวข้อหลัก > หัวข้อย่อย]
  เนื้อหาในข้อความ...
  ```
* **Overlap:** กำหนดจุดเหลื่อมซ้อน (เช่น 150 ตัวอักษร) เพื่อป้องกันความหมายขาดตอนระหว่าง Chunk

### Step 3: Embedding Generation (สร้างเวกเตอร์ความหมาย)
* นำ Chunk แต่ละชิ้นส่งให้ **Embedding Model** ประมวลผลเป็น Dense Vector (เช่น เวกเตอร์ขนาด 1024 มิติ)
* ทำการ Normalize เวกเตอร์ (L2-Normalization) เพื่อให้สามารถคำนวณ Cosine Similarity ได้รวดเร็วผ่าน Matrix Dot Product (`A @ B`)

### Step 4: Storage & Indexing (การจัดเก็บและระบบแคช)
* บันทึกข้อมูลแบ่งเป็น 2 ส่วน:
  1. **Vectors Array:** เก็บค่าตัวเลขเวกเตอร์
  2. **Metadata JSON:** เก็บรายละเอียดชิ้นข้อความ, ชื่อไฟล์ต้นทาง, และ Path หัวข้อ
* **Smart Incremental Re-indexing (คำนวณ Hash SHA-256):** ตรวจสอบ Hash ของไฟล์ หากไฟล์ไม่มีการแก้ไข จะดึงจาก Cache มาใช้ทันที โดยอัปเดตเฉพาะไฟล์ที่มีการเพิ่ม/ลบ/แก้ไข

### Step 5: Retrieval & Generation (การค้นหาและสร้างคำตอบ)
1. เมื่อผู้ใช้ส่งคำถาม (Query) ➡️ แปลงคำถามเป็น Vector ด้วย Embedding Model ตัวเดียวกัน
2. คำนวณความเหมือนทางความหมาย (Cosine Similarity Score)
3. คัดเลือกเฉพาะ Chunk ที่มี Score สูงกว่า Threshold (เช่น `min_score = 0.35`) จำนวน `top_k` ชิ้น
4. นำข้อความบริบท (Context) ไปแทรกใน System Prompt ของ LLM

---

## 🛠️ 3. เครื่องมือและไลบรารีที่นิยมใช้ (Tech Stack)

| หน้าที่ | เครื่องมือ/ไลบรารีที่นิยมใช้ |
| :--- | :--- |
| **RAG Frameworks** | `LangChain`, `LlamaIndex`, `Haystack` |
| **Markdown Chunking** | `MarkdownHeaderTextSplitter` (LangChain), `mistune`, Custom Regex |
| **Embedding Models** | `BAAI/bge-m3`, `sentence-transformers/all-MiniLM-L6-v2`, `text-embedding-3-small` (OpenAI) |
| **Vector DB / Storage** | `ChromaDB`, `FAISS`, `Qdrant`, `LanceDB` หรือ **NumPy (`.npy`)** |
| **LLM Engine** | OpenAI API, Ollama, vLLM, LM Studio |

---

## 💡 4. กรณีศึกษา: สถาปัตยกรรม RAG ในโปรเจกต์นี้ (`Sci-VTuberLLM`)

ใน codebase นี้มีการสร้างระบบ **Lightweight Native Markdown RAG** โดยไม่จำเป็นต้องติดตั้ง Vector Database ภายนอก:

1. **[chunker.py](file:///c:/Users/jirathx/Desktop/Sci-VTuberLLM/src/open_llm_vtuber/knowledge/chunker.py)**: ใช้ Regex สแกน `#` ตัดข้อความตามโครงสร้าง Markdown และแปะ `[file > heading]`
2. **[embedder.py](file:///c:/Users/jirathx/Desktop/Sci-VTuberLLM/src/open_llm_vtuber/knowledge/embedder.py)**: ใช้ HuggingFace Transformers โหลดโมเดล `BAAI/bge-m3` ทำ Mean Pooling & L2 Normalization
3. **[knowledge_base.py](file:///c:/Users/jirathx/Desktop/Sci-VTuberLLM/src/open_llm_vtuber/knowledge/knowledge_base.py)**: เก็บ Vector ใน `vectors.npy` และ Metadata ใน `index.json` พร้อมระบบ Hash SHA-256 ตรวจจับการเปลี่ยนแปลงของไฟล์ในโฟลเดอร์ [knowledge_md](file:///c:/Users/jirathx/Desktop/Sci-VTuberLLM/knowledge_md)
4. **[single_conversation.py](file:///c:/Users/jirathx/Desktop/Sci-VTuberLLM/src/open_llm_vtuber/conversations/single_conversation.py#L74-L90)**: ทำ Cosine Retrieval ดึง Top-K Chunks แล้วส่งต่อไปยัง LLM Prompt เพื่อบังคับให้บอทตอบคำถามโดยอิงจากเอกสาร

---

## 🚀 5. คำแนะนำเพื่อปรับปรุงประสิทธิภาพ RAG
1. **รักษาโครงสร้าง Heading:** เขียนไฟล์ Markdown โดยใช้ `#`, `##` อย่างเป็นระเบียบ จะช่วยให้ Chunking แบ่งบริบทได้แม่นยำที่สุด
2. **ปรับแต่ง Chunk Size & Overlap:** สำหรับภาษาไทย แนะนำ Chunk Size อยู่ที่ประมาณ 500-1000 ตัวอักษร
3. **ใช้ Embedding Model ที่รองรับหลายภาษา:** โมเดลอย่าง `BAAI/bge-m3` รองรับทั้งภาษาไทยและภาษาอังกฤษได้เป็นอย่างดี
