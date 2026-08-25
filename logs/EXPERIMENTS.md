# สมุดบันทึกการทดลอง

บันทึกอัตโนมัติจาก `src/journal.py` — ทุกครั้งที่รัน pipeline หรือ benchmark
จะต่อท้ายไฟล์นี้ ไม่มีการเขียนทับ เรียงจากเก่าไปใหม่

## ✅ 2026-08-25 18:44 — setup: GPU venv

- **ทดลองอะไร:** ตั้ง venv แยกที่มี CUDA เพราะ env หลักเป็น torch 2.13.0+cpu ทำให้ RTX 5050 ไม่ถูกใช้
- **ทำอย่างไร:** python -m venv .venv-gpu; pip install torch==2.11.0+cu128 --index-url .../cu128; ลง dependency ที่เหลือให้ version ตรงกับ env หลักทุกตัว
- **ผลลัพธ์:** CUDA ใช้ได้จริง RTX 5050 VRAM 8.55GB compute capability 12.0, matmul 2048x2048 ผ่าน

```json
{
  "torch_gpu": "2.11.0+cu128",
  "torch_cpu": "2.13.0+cpu",
  "vram_gb": 8.55
}
```

## ✅ 2026-08-25 18:47 — pipeline steps 1-2: bge-m3

- **ทดลองอะไร:** รัน pipeline ขั้น 1-2 ด้วย embedding model bge-m3
- **ทำอย่างไร:** 9 ไฟล์จาก ['curriculum'], chunk_size=150, overlap=30, max_seq=512, device=auto
- **ผลลัพธ์:** รวม 16.9s, ครบทุกขั้น

```json
{
  "model": "bge-m3",
  "hf_id": "BAAI/bge-m3",
  "step_seconds": {
    "1/7 แยกข้อความจาก PDF": 5.03,
    "2/7 ตัด chunk": 10.27
  },
  "total_seconds": 16.93,
  "failed_steps": []
}
```

## ✅ 2026-08-25 18:49 — pipeline steps 1: bge-m3

- **ทดลองอะไร:** รัน pipeline ขั้น 1 ด้วย embedding model bge-m3
- **ทำอย่างไร:** 9 ไฟล์จาก ['curriculum'], chunk_size=150, overlap=30, max_seq=512, device=auto
- **ผลลัพธ์:** รวม 6.7s, ครบทุกขั้น

```json
{
  "model": "bge-m3",
  "hf_id": "BAAI/bge-m3",
  "step_seconds": {
    "1/7 แยกข้อความจาก PDF": 5.26
  },
  "total_seconds": 6.69,
  "failed_steps": []
}
```

## ✅ 2026-08-25 18:51 — ตั้งค่า max_seq_length จากการวัดจริง

- **ทดลองอะไร:** ต้องตัดสินใจว่าจะจำกัดความยาว sequence เท่าไหร่ ค่า default ของ Qwen3 คือ 32k ซึ่งเกินความจำเป็นมาก และ attention เป็น O(n^2)
- **ทำอย่างไร:** นับ token ของ chunk ทั้ง 3,128 อันด้วย tokenizer ของ bge-m3 (XLM-R), e5-base และ Qwen3-0.6B
- **ผลลัพธ์:** ตั้ง 768 — ไม่ตัดข้อความทิ้งเลยสักอัน (Qwen3 ยาวสุด 699 token, XLM-R ยาวสุด 271) ถ้าตั้ง 512 จะตัด 10 chunks เฉพาะตระกูล Qwen3

```json
{
  "chunks": 3128,
  "xlmr": {
    "p50": 61,
    "p90": 158,
    "p95": 172,
    "p99": 200,
    "max": 271,
    "over_512": 0
  },
  "qwen3": {
    "p50": 109,
    "p90": 325,
    "p95": 361,
    "p99": 418,
    "max": 699,
    "over_512": 10
  },
  "chosen_max_seq": 768,
  "note": "e5-base ถูกจำกัดที่ 512 อยู่แล้วเพราะมี position embedding แค่ 514 ตัว"
}
```

## ❌ 2026-08-25 18:51 — ปัญหาข้อมูล: PDF 2 ไฟล์ดึงข้อความไม่ออก

- **ทดลองอะไร:** ตรวจว่าไฟล์ใหม่ 6 ไฟล์ที่เพิ่มเข้ามาใช้งานได้จริงหรือไม่
- **ทำอย่างไร:** เทียบขนาดไฟล์กับจำนวนตัวอักษรที่ดึงได้ ในขั้นที่ 1 ของ pipeline
- **ผลลัพธ์:** Regulations-regulations-announcements-2025.pdf (54 MB, 76 หน้า) ได้ 5,173 ตัวอักษร และ Regulations-rules-and-announcements_2026-05-19.pdf (42 MB, 76 หน้า) ได้ 4,136 ตัวอักษร ทั้งคู่เป็นสแกน ต้องใช้ OCR — เข้า index แล้วแต่แทบไม่มีเนื้อหา

```json
{
  "files": [
    {
      "name": "Regulations-regulations-announcements-2025.pdf",
      "kb": 54268,
      "chars": 5173,
      "chunks": 13
    },
    {
      "name": "Regulations-rules-and-announcements_2026-05-19.pdf",
      "kb": 42625,
      "chars": 4136,
      "chunks": 37
    }
  ]
}
```

## ✅ 2026-08-25 18:54 — benchmark embedding: e5-base

- **ทดลองอะไร:** วัดความเร็วและคุณภาพของ intfloat/multilingual-e5-base (278M, มิติ 768)
- **ทำอย่างไร:** เข้ารหัส 3128 chunks บน cuda, batch=32, max_seq=512; ประเมิน dense-only ด้วย golden set 128 คำถาม
- **ผลลัพธ์:** โหลด 11.03s, เข้ารหัส 18.4s (170.04 chunks/s), ต่อคำถาม p50 12.29ms, Recall@1 72.7%, MRR 0.803

```json
{
  "model_key": "e5-base",
  "hf_id": "intfloat/multilingual-e5-base",
  "params_m": 278,
  "dim": 768,
  "max_seq_length": 512,
  "device": "cuda",
  "batch_size": 32,
  "n_chunks": 3128,
  "n_queries": 128,
  "load_seconds": 11.03,
  "encode_seconds": 18.4,
  "chunks_per_second": 170.04,
  "index_seconds": 0.078,
  "query_ms_mean": 14.5,
  "query_ms_p50": 12.29,
  "query_ms_p95": 23.94,
  "embeddings_mb": 9.2,
  "rss_before_gb": 0.5,
  "rss_after_gb": 1.75,
  "measured_at": "2026-08-25T18:54:39",
  "quality": {
    "all": {
      "mrr": 0.803351314484127,
      "hit@1": 0.7265625,
      "recall@1": 0.7265625,
      "ndcg@1": 0.7265625,
      "hit@3": 0.859375,
      "recall@3": 0.859375,
      "ndcg@3": 0.8052434143973242,
      "hit@5": 0.890625,
      "recall@5": 0.890625,
      "ndcg@5": 0.8187020568371177,
      "hit@10": 0.9609375,
      "recall@10": 0.9609375,
      "ndcg@10": 0.8411341036043546
    },
    "by_name": {
      "mrr": 0.9609375,
      "hit@1": 0.921875,
      "recall@1": 0.921875,
      "ndcg@1": 0.921875,
      "hit@3": 1.0,
      "recall@3": 1.0,
      "ndcg@3": 0.9711663869977701,
      "hit@5": 1.0,
      "recall@5": 1.0,
      "ndcg@5": 0.9711663869977701,
      "hit@10": 1.0,
      "recall@10": 1.0,
      "ndcg@10": 0.9711663869977701
    },
    "by_code": {
      "mrr": 0.645765128968254,
      "hit@1": 0.53125,
      "recall@1": 0.53125,
      "ndcg@1": 0.53125,
      "hit@3": 0.71875,
      "recall@3": 0.71875,
      "ndcg@3": 0.6393204417968782,
      "hit@5": 0.78125,
      "recall@5": 0.78125,
      "ndcg@5": 0.6662377266764652,
      "hit@10": 0.921875,
      "recall@10": 0.921875,
      "ndcg@10": 0.711101820210939
    }
  }
}
```

## ✅ 2026-08-25 18:56 — benchmark embedding: bge-m3

- **ทดลองอะไร:** วัดความเร็วและคุณภาพของ BAAI/bge-m3 (568M, มิติ 1024)
- **ทำอย่างไร:** เข้ารหัส 3128 chunks บน cuda, batch=32, max_seq=768; ประเมิน dense-only ด้วย golden set 128 คำถาม
- **ผลลัพธ์:** โหลด 11.38s, เข้ารหัส 58.44s (53.53 chunks/s), ต่อคำถาม p50 25.65ms, Recall@1 63.3%, MRR 0.707

```json
{
  "model_key": "bge-m3",
  "hf_id": "BAAI/bge-m3",
  "params_m": 568,
  "dim": 1024,
  "max_seq_length": 768,
  "device": "cuda",
  "batch_size": 32,
  "n_chunks": 3128,
  "n_queries": 128,
  "load_seconds": 11.38,
  "encode_seconds": 58.44,
  "chunks_per_second": 53.53,
  "index_seconds": 0.08,
  "query_ms_mean": 28.4,
  "query_ms_p50": 25.65,
  "query_ms_p95": 40.16,
  "embeddings_mb": 12.2,
  "rss_before_gb": 0.5,
  "rss_after_gb": 1.72,
  "measured_at": "2026-08-25T18:56:33",
  "quality": {
    "all": {
      "mrr": 0.7066282242063492,
      "hit@1": 0.6328125,
      "recall@1": 0.6328125,
      "ndcg@1": 0.6328125,
      "hit@3": 0.75,
      "recall@3": 0.75,
      "ndcg@3": 0.6995893595982161,
      "hit@5": 0.8125,
      "recall@5": 0.8125,
      "ndcg@5": 0.724452406157232,
      "hit@10": 0.8984375,
      "recall@10": 0.8984375,
      "ndcg@10": 0.7518128680205951
    },
    "by_name": {
      "mrr": 0.9791666666666666,
      "hit@1": 0.96875,
      "recall@1": 0.96875,
      "ndcg@1": 0.96875,
      "hit@3": 0.984375,
      "recall@3": 0.984375,
      "ndcg@3": 0.978608277399554,
      "hit@5": 0.984375,
      "recall@5": 0.984375,
      "ndcg@5": 0.978608277399554,
      "hit@10": 1.0,
      "recall@10": 1.0,
      "ndcg@10": 0.9841740146981168
    },
    "by_code": {
      "mrr": 0.43408978174603174,
      "hit@1": 0.296875,
      "recall@1": 0.296875,
      "ndcg@1": 0.296875,
      "hit@3": 0.515625,
      "recall@3": 0.515625,
      "ndcg@3": 0.4205704417968782,
      "hit@5": 0.640625,
      "recall@5": 0.640625,
      "ndcg@5": 0.47029653491490997,
      "hit@10": 0.796875,
      "recall@10": 0.796875,
      "ndcg@10": 0.5194517213430734
    }
  }
}
```

## ❌ 2026-08-25 18:59 — benchmark embedding: pixie-rune

- **ทดลองอะไร:** วัดความเร็วและคุณภาพของ telepix/PIXIE-Rune-v1.0
- **ทำอย่างไร:** device=cuda, max_seq=768, chunks=3128
- **ผลลัพธ์:** ล้มเหลว: RuntimeError: Cannot send a request, as the client has been closed.

```json
{
  "model": "pixie-rune",
  "error_type": "RuntimeError"
}
```

## ❌ 2026-08-25 18:59 — benchmark embedding: octen-0.6b

- **ทดลองอะไร:** วัดความเร็วและคุณภาพของ bflhc/Octen-Embedding-0.6B
- **ทำอย่างไร:** device=cuda, max_seq=768, chunks=3128
- **ผลลัพธ์:** ล้มเหลว: RuntimeError: Cannot send a request, as the client has been closed.

```json
{
  "model": "octen-0.6b",
  "error_type": "RuntimeError"
}
```

## ❌ 2026-08-25 18:59 — benchmark embedding: embeddinggemma-300m

- **ทดลองอะไร:** วัดความเร็วและคุณภาพของ google/embeddinggemma-300m
- **ทำอย่างไร:** device=cuda, max_seq=768, chunks=3128
- **ผลลัพธ์:** ล้มเหลว: GatedRepoError: 401 Client Error. (Request ID: Root=1-6a8d8397-4fb5543d6db0d1567494d7f7;aa442362-3588-47e6-8e77-14cdb5b5a787)

Cannot access gated repo for url https://huggingface.co/google/embeddinggemma-300m/resolve/main/modules.json.
Access to model google/embeddinggemma-300m is restricted. You must have access 

```json
{
  "model": "embeddinggemma-300m",
  "error_type": "GatedRepoError"
}
```

## ❌ 2026-08-25 19:00 — บั๊ก: huggingface_hub retry ใช้ httpx client ที่ปิดไปแล้ว

- **ทดลองอะไร:** benchmark 5 โมเดล แล้ว pixie-rune กับ octen-0.6b ล้มเหลวทั้งคู่ตอนโหลดโมเดล
- **ทำอย่างไร:** เรียก SentenceTransformer() ผ่าน .venv-gpu (huggingface-hub 1.24.0) ดาวน์โหลดครั้งแรก
- **ผลลัพธ์:** HF Hub ตัดการเชื่อมต่อ (WinError 10054) ระหว่างขอ processor_config.json แล้ว retry ภายในของ huggingface_hub เจอ RuntimeError: Cannot send a request, as the client has been closed — retry path ของ library ใช้ httpx client ตัวเดิมที่ถูกปิดไปแล้ว retry 5 ครั้งของมันจึงไร้ผล แก้โดยเพิ่ม retry รอบนอกใน EmbeddingModel.load() ที่สร้าง SentenceTransformer ใหม่ทั้งก้อน (ได้ client ใหม่) เฉพาะ error ที่เป็น transport เท่านั้น

```json
{
  "affected": [
    "pixie-rune",
    "octen-0.6b"
  ],
  "library": "huggingface-hub 1.24.0",
  "fix": "src/embedding_model.py: load(attempts=3) + exponential sleep"
}
```

## ❌ 2026-08-25 19:00 — embeddinggemma-300m โหลดไม่ได้ (gated)

- **ทดลองอะไร:** ลองโหลด google/embeddinggemma-300m ซึ่งเป็นอันดับ 2 ของ Thai-MTEB
- **ทำอย่างไร:** SentenceTransformer(google/embeddinggemma-300m) โดยไม่มี HF token
- **ผลลัพธ์:** GatedRepoError 401 — Access to model google/embeddinggemma-300m is restricted ต้องกดยอมรับ licence ที่หน้า HF แล้วตั้ง HF_TOKEN จึงจะทดสอบได้ ไม่ใช่ปัญหาของเครื่องหรือโค้ด

```json
{
  "blocker": "ต้องมี HF token + ยอมรับ licence"
}
```

## ✅ 2026-08-25 19:01 — benchmark embedding: pixie-rune

- **ทดลองอะไร:** วัดความเร็วและคุณภาพของ telepix/PIXIE-Rune-v1.0 (568M, มิติ 1024)
- **ทำอย่างไร:** เข้ารหัส 3128 chunks บน cuda, batch=32, max_seq=768; ประเมิน dense-only ด้วย golden set 128 คำถาม
- **ผลลัพธ์:** โหลด 17.11s, เข้ารหัส 56.41s (55.46 chunks/s), ต่อคำถาม p50 24.89ms, Recall@1 62.5%, MRR 0.703

```json
{
  "model_key": "pixie-rune",
  "hf_id": "telepix/PIXIE-Rune-v1.0",
  "params_m": 568,
  "dim": 1024,
  "max_seq_length": 768,
  "device": "cuda",
  "batch_size": 32,
  "n_chunks": 3128,
  "n_queries": 128,
  "load_seconds": 17.11,
  "encode_seconds": 56.41,
  "chunks_per_second": 55.46,
  "index_seconds": 0.149,
  "query_ms_mean": 27.26,
  "query_ms_p50": 24.89,
  "query_ms_p95": 40.32,
  "embeddings_mb": 12.2,
  "rss_before_gb": 0.5,
  "rss_after_gb": 1.74,
  "measured_at": "2026-08-25T19:01:53",
  "quality": {
    "all": {
      "mrr": 0.7033203125,
      "hit@1": 0.625,
      "recall@1": 0.625,
      "ndcg@1": 0.625,
      "hit@3": 0.734375,
      "recall@3": 0.734375,
      "ndcg@3": 0.6929850530971011,
      "hit@5": 0.796875,
      "recall@5": 0.796875,
      "ndcg@5": 0.7192175918698311,
      "hit@10": 0.875,
      "recall@10": 0.875,
      "ndcg@10": 0.7441615664412111
    },
    "by_name": {
      "mrr": 0.953125,
      "hit@1": 0.90625,
      "recall@1": 0.90625,
      "ndcg@1": 0.90625,
      "hit@3": 1.0,
      "recall@3": 1.0,
      "ndcg@3": 0.9653996643973242,
      "hit@5": 1.0,
      "recall@5": 1.0,
      "ndcg@5": 0.9653996643973242,
      "hit@10": 1.0,
      "recall@10": 1.0,
      "ndcg@10": 0.9653996643973242
    },
    "by_code": {
      "mrr": 0.453515625,
      "hit@1": 0.34375,
      "recall@1": 0.34375,
      "ndcg@1": 0.34375,
      "hit@3": 0.46875,
      "recall@3": 0.46875,
      "ndcg@3": 0.4205704417968782,
      "hit@5": 0.59375,
      "recall@5": 0.59375,
      "ndcg@5": 0.4730355193423382,
      "hit@10": 0.75,
      "recall@10": 0.75,
      "ndcg@10": 0.5229234684850981
    }
  }
}
```
