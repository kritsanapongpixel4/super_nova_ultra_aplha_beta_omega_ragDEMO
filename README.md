# 🚀 Super Nova Ultra Alpha Beta Omega RAG DEMO

> **3 RMUTT CPE students who if not finish this project they can't graduated 💀**

ระบบ **Retrieval-Augmented Generation (RAG)** สำหรับตอบคำถามเกี่ยวกับเอกสารของมหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี (RMUTT) ภาควิชาวิศวกรรมคอมพิวเตอร์ — รวมถึงแบบฟอร์มต่าง ๆ, หลักสูตร, CLOs, ข้อกำหนดสหกิจ, ตำราเรียน และอื่น ๆ

---

## 📐 สถาปัตยกรรมระบบ (Architecture)

```
                        ┌────────────────────┐
                        │   📂 data/          │  ← PDF / TXT / DOCX / MD
                        └────────┬───────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  1. Extract Text         │  pipeline/extract_text.py
                    │  (PyMuPDF อ่าน PDF)      │  → outputs/extracted_text.json
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  2. Chunking             │  pipeline/chunking.py
                    │  (PyThaiNLP + Recursive) │  → outputs/chunks.json
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  3. Embedding            │  pipeline/create_embeddings.py
                    │  (BAAI/bge-m3)           │  → outputs/embeddings.npy
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                   │
    ┌─────────▼─────────┐  ┌────▼─────────┐  ┌──────▼──────────┐
    │  4a. FAISS Index   │  │ 4b. BM25     │  │ 4c. Chunk Store │
    │  (Dense Search)    │  │ (Sparse)     │  │ (JSON)          │
    └─────────┬─────────┘  └────┬─────────┘  └──────┬──────────┘
              │                  │                   │
              └──────────────────┼───────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  5. Hybrid Retrieval     │  src/hybrid_retriever.py
                    │  (RRF Fusion)            │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  6. Cross-Encoder        │  src/rerankers.py
                    │  Reranking (bge-reranker)│
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  7. LLM Generation       │  src/generator.py
                    │  (Claude Opus 5)         │
                    └────────────┴────────────┘
                                 │
                           ✅ คำตอบ + แหล่งอ้างอิง
```

---

## 🗂️ โครงสร้างโปรเจกต์

```
super_nova_ultra_aplha_beta_omega_ragDEMO/
├── config.py                    # ⚙️  ค่า config ทั้งหมด (paths, models, params)
├── main.py                      # 🚪 Entry point — chat / ถามคำถามเดี่ยว
├── build_index.py               # 🔨 สร้าง index ทั้งหมดในครั้งเดียว
│
├── data/                        # 📂 ไฟล์ต้นฉบับ (29 PDF)
│   ├── 01-แบบคำร้องทั่วไป.pdf
│   ├── หลักสูตร-683.pdf
│   ├── CLOs-Computer_Engineering-RMUTT.pdf
│   ├── การสื่อสารข้อมูลA4-2_pub.pdf
│   └── ... (รวม 29 ไฟล์)
│
├── pipeline/                    # 🔄 แต่ละ step รันแยกได้
│   ├── extract_text.py          # Step 1: ดึงข้อความจาก PDF
│   ├── chunking.py              # Step 2: แบ่ง chunk (Thai-aware)
│   ├── create_embeddings.py     # Step 3: สร้าง embeddings
│   ├── create_vector_db.py      # Step 4: สร้าง FAISS + BM25 index
│   ├── query_embedding.py       # Step 5: embed คำถาม
│   ├── similarity_search.py     # Step 6: ค้นหา
│   └── complete_retrieval.py    # Step 7: retrieval + rerank
│
├── src/                         # 📦 Core modules
│   ├── document_loader.py       # อ่านไฟล์ PDF/TXT/DOCX/MD
│   ├── text_splitter.py         # Thai chunking (PyThaiNLP + LangChain)
│   ├── embedding_model.py       # wrapper สำหรับ embedding model
│   ├── vector_store.py          # FAISS index management
│   ├── hybrid_retriever.py      # BM25 + Dense + RRF fusion
│   ├── rerankers.py             # Cross-encoder reranking
│   ├── generator.py             # Claude LLM generation
│   ├── prompt_templates.py      # Prompt engineering
│   ├── memory.py                # Conversation history
│   ├── query_transform.py       # Query rewriting
│   ├── retriever.py             # Base retriever interface
│   ├── index_meta.py            # Index freshness tracking
│   └── rag_pipeline.py          # End-to-end pipeline orchestrator
│
├── evaluation/                  # 📊 Evaluation tools
│   ├── build_golden_set.py      # สร้าง golden Q&A set
│   ├── eval_retrieval.py        # ประเมิน retrieval (Recall@K, MRR)
│   ├── eval_generation.py       # ประเมิน generation quality
│   └── metrics.py               # Metric functions
│
├── outputs/                     # 📤 ผลลัพธ์ระหว่าง pipeline
│   ├── extracted_text.json
│   ├── chunks.json
│   └── embeddings.npy
│
└── vector_db/                   # 💾 Indexes ที่ build แล้ว
    ├── document.index           # FAISS
    ├── bm25_index.pkl           # BM25
    ├── chunk_store.json         # chunk metadata
    └── index_meta.json          # fingerprint
```

---

## 🧠 ทำไมถึงเลือก Model เหล่านี้?

### Embedding: `BAAI/bge-m3` (dim=1024)

| เหตุผล | รายละเอียด |
|--------|-----------|
| 🌏 **Multilingual** | รองรับ 100+ ภาษา รวมถึง **ภาษาไทย** โดยตรง — ไม่ต้อง fine-tune เพิ่ม |
| 📏 **Multi-Granularity** | M3 = Multi-lingual, Multi-Functionality, Multi-Granularity สามารถ embed ได้ทั้งประโยคสั้นและย่อหน้ายาว ซึ่งเหมาะกับข้อมูลของเราที่มีตั้งแต่ช่องกรอกฟอร์มสั้น ๆ ไปจนถึงเนื้อหาตำราเรียน |
| 🏆 **Performance** | ติด Top MTEB leaderboard สำหรับ multilingual tasks |
| 🔄 **Dense + Sparse** | bge-m3 รองรับทั้ง dense embedding และ sparse (lexical) ในโมเดลเดียว ซึ่งช่วย hybrid retrieval |
| 📐 **1024 dim** | มิติสูงพอสำหรับ semantic nuance แต่ไม่ใหญ่เกินจนช้า |
| 🆓 **Open-source** | ใช้ผ่าน `sentence-transformers` ได้ฟรี ไม่ต้องจ่ายค่า API |

**ทำไมไม่ใช้ตัวอื่น?**
- `text-embedding-ada-002` (OpenAI) → เสียเงิน + ไม่ค่อยแม่นกับภาษาไทย
- `multilingual-e5-large` → ดีกับไทย แต่ bge-m3 มี sparse mode ด้วย
- `WangchanBERTa` → เฉพาะไทย ไม่รองรับ mixed Thai-English content ดี

### Reranker: `BAAI/bge-reranker-v2-m3`

| เหตุผล | รายละเอียด |
|--------|-----------|
| 🎯 **Cross-Encoder** | อ่าน query + document พร้อมกัน → แม่นกว่า bi-encoder (embedding) มาก |
| 🤝 **คู่กับ bge-m3** | ออกแบบมาคู่กันจาก BAAI — embedding ดึงแบบหยาบ, reranker จัดลำดับแบบละเอียด |
| 🌏 **Multilingual** | รองรับภาษาไทยเหมือน bge-m3 |

### LLM Generation: `Claude Opus 5`

| เหตุผล | รายละเอียด |
|--------|-----------|
| 📚 **Long Context** | รองรับ context ยาว → ใส่ chunks ได้เยอะ |
| 🇹🇭 **ภาษาไทยดี** | เข้าใจและตอบภาษาไทยได้เป็นธรรมชาติ |
| 🎯 **Instruction Following** | ทำตาม prompt template ได้แม่นยำ — ไม่แต่งคำตอบเอง |

---

## ✂️ ระบบ Chunking (Thai-Aware)

เราใช้ **PyThaiNLP + LangChain RecursiveCharacterTextSplitter** แทนการตัดแบบ character ธรรมดา:

```python
# chunk_size วัดเป็น "จำนวนคำไทย" ไม่ใช่ตัวอักษร
from pythainlp.tokenize import word_tokenize  # engine="newmm"
```

**ค่าที่ tune แล้ว (จากการวิเคราะห์ข้อมูลจริง):**

| Parameter | ค่า | เหตุผล |
|-----------|-----|--------|
| `CHUNK_SIZE` | **150 tokens** | 73.5% ของ records < 50 tokens, median = 18 → ค่าเล็กรักษา granularity |
| `CHUNK_OVERLAP` | **30 tokens** | 20% overlap เพียงพอสำหรับความต่อเนื่อง |

**Separators (ลำดับความสำคัญ):**
`\n\n` → `\n` → `space` → `ๆ` → character-level fallback

---

## 🔍 Hybrid Retrieval Strategy

```
Query → ┌─── Dense Search (FAISS + bge-m3) ───── top 20 candidates
         │
         └─── Sparse Search (BM25 + PyThaiNLP tokenize) ── top 20 candidates
                                    │
                    ┌───────────────▼───────────────┐
                    │  Reciprocal Rank Fusion (RRF)  │
                    │  score = Σ 1/(60 + rank)       │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  Cross-Encoder Reranking       │
                    │  (bge-reranker-v2-m3)          │
                    └───────────────┬───────────────┘
                                    │
                              Top 5 chunks → LLM
```

**ทำไมต้อง Hybrid?**
- **Dense (embedding)** จับ _ความหมาย_ ได้ดี — ถามว่า "ขั้นตอนลาป่วย" จะเจอ "การลากิจส่วนตัว"
- **BM25 (keyword)** จับ _คำเฉพาะ_ ได้ดี — ถามว่า "ศรม.13" จะเจอฟอร์มตรง ๆ
- **RRF** รวม ranking ทั้งสองแบบ โดยไม่ต้องทำให้ score scale เท่ากัน

---

## ⚡ Quick Start

### 1. ติดตั้ง Dependencies

```bash
pip install pythainlp pymupdf langchain-text-splitters sentence-transformers faiss-cpu anthropic rank-bm25
```

### 2. สร้าง Index (ครั้งแรก / เมื่อเพิ่มเอกสาร)

```bash
# รันทีละ step เพื่อดูผลแต่ละขั้น
python pipeline/extract_text.py     # 1. ดึงข้อความจาก PDF
python pipeline/chunking.py         # 2. แบ่ง chunk
python pipeline/create_embeddings.py # 3. สร้าง embeddings
python pipeline/create_vector_db.py  # 4. สร้าง FAISS + BM25

# หรือรันทีเดียว
python build_index.py
```

### 3. ถามคำถาม

```bash
# คำถามเดียว
python main.py "ขั้นตอนการลาพักการศึกษาเป็นอย่างไร?"

# Interactive chat
python main.py
```

---

## 📊 สถิติข้อมูล (Data Stats)

| รายการ | จำนวน |
|--------|-------|
| ไฟล์ทั้งหมดใน `data/` | 29 PDF |
| ไฟล์ที่ extract ได้ | 26 ไฟล์ (3 ไฟล์เป็น scanned image) |
| Records ที่ได้ | 5,304 records |
| Median token length | 18 tokens/record |
| ไฟล์ใหญ่สุด | การสื่อสารข้อมูล (2,011 records) |

---

## 📋 สถานะการพัฒนา

| Module | สถานะ | หมายเหตุ |
|--------|--------|---------|
| `document_loader.py` | ✅ เสร็จ | รองรับ PDF/TXT/DOCX/MD |
| `text_splitter.py` | ✅ เสร็จ | PyThaiNLP + Recursive |
| `extract_text.py` | ✅ เสร็จ | Auto-discover ทุกไฟล์ |
| `chunking.py` | ✅ เสร็จ | Thai-aware chunking |
| `embedding_model.py` | 🚧 TODO | wrapper skeleton อยู่ |
| `vector_store.py` | 🚧 TODO | FAISS wrapper |
| `hybrid_retriever.py` | 🚧 TODO | BM25 + Dense + RRF |
| `rerankers.py` | 🚧 TODO | Cross-encoder |
| `generator.py` | ✅ เสร็จ | Claude Opus 5 |
| `prompt_templates.py` | ✅ เสร็จ | Grounded answering |
| `rag_pipeline.py` | 🚧 TODO | Orchestrator |
| `main.py` | 🚧 TODO | Chat interface |
| `build_index.py` | 🚧 TODO | Full pipeline runner |
| Evaluation | 🚧 TODO | Retrieval + Generation eval |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.14 |
| PDF Extraction | PyMuPDF |
| Thai Tokenization | PyThaiNLP (newmm engine) |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embedding | BAAI/bge-m3 (via sentence-transformers) |
| Vector DB | FAISS |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranking | BAAI/bge-reranker-v2-m3 |
| LLM | Claude Opus 5 (Anthropic API) |
| Fusion | Reciprocal Rank Fusion (RRF) |

---

## 📝 Config ทั้งหมด (อยู่ใน `config.py`)

ทุกค่าที่ tune ได้อยู่ใน [`config.py`](config.py) ที่เดียว — ไม่ต้องไปหาตาม source code:

```python
CHUNK_SIZE = 150           # Thai tokens per chunk
CHUNK_OVERLAP = 30         # overlap between chunks
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
TOP_K = 5                  # chunks ส่งให้ LLM
CANDIDATE_K = 20           # chunks ดึงก่อน rerank
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "claude-opus-5"
```

---

*พัฒนาโดยนักศึกษา CPE, RMUTT — 67 69 นน เกย์ 💀*
