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

## ❌ 2026-08-25 19:07 — benchmark embedding: octen-0.6b

- **ทดลองอะไร:** วัดความเร็วและคุณภาพของ bflhc/Octen-Embedding-0.6B
- **ทำอย่างไร:** device=cuda, max_seq=768, chunks=3128
- **ผลลัพธ์:** ล้มเหลว: ValueError: Unrecognized processing class in bflhc/Octen-Embedding-0.6B. Can't instantiate a processor, a tokenizer, an image processor, a video processor or a feature extractor for this model. Make sure the repository contains the files of at least one of those processing classes.

```json
{
  "model": "octen-0.6b",
  "error_type": "ValueError"
}
```

## ✅ 2026-08-25 19:11 — benchmark LLM API latency

- **ทดลองอะไร:** วัดเวลาตอบกลับของโมเดลภาษาแต่ละตัว (TTFT, เวลารวม, tokens/วินาที)
- **ทำอย่างไร:** 3 คำถามจริง × 2 ครั้ง/โมเดล, prompt สร้างจาก hybrid retrieval บน index ของ bge-m3, เรียกแบบ streaming ไม่มี retry
- **ผลลัพธ์:** gemini-3.5-flash: TTFT 3.908s รวม 4.192s, gemini-3.6-flash: TTFT 9.518s รวม 9.933s, gemini-3.7-flash: TTFT 75.144s รวม 75.189s, gemini-3-flash-preview: TTFT 7.366s รวม 7.737s, gemini-3.1-flash-lite: TTFT 3.369s รวม 3.472s

```json
{
  "providers_with_keys": {
    "gemini": true,
    "openai": false,
    "anthropic": false
  },
  "summary": {
    "gemini-3.5-flash": {
      "calls": 6,
      "ok": 6,
      "failed": 0,
      "errors": [],
      "ttft_s": {
        "mean": 3.908,
        "min": 2.158,
        "max": 6.862
      },
      "total_s": {
        "mean": 4.192,
        "min": 2.163,
        "max": 7.073
      },
      "tokens_per_s": {
        "mean": 3792.017,
        "min": 307.3,
        "max": 13558.5
      },
      "input_tokens": {
        "mean": 1290.333,
        "min": 1153,
        "max": 1403
      },
      "output_tokens": {
        "mean": 128.167,
        "min": 33,
        "max": 209
      }
    },
    "gemini-3.6-flash": {
      "calls": 6,
      "ok": 6,
      "failed": 0,
      "errors": [],
      "ttft_s": {
        "mean": 9.518,
        "min": 4.486,
        "max": 23.583
      },
      "total_s": {
        "mean": 9.933,
        "min": 4.487,
        "max": 24.486
      },
      "tokens_per_s": {
        "mean": 3698.967,
        "min": 129.7,
        "max": 20706.6
      },
      "input_tokens": {
        "mean": 1290.333,
        "min": 1153,
        "max": 1403
      },
      "output_tokens": {
        "mean": 116.167,
        "min": 32,
        "max": 219
      }
    },
    "gemini-3.7-flash": {
      "calls": 6,
      "ok": 3,
      "failed": 3,
      "errors": [
        "ReadTimeout: The read operation timed out",
        "ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
      ],
      "ttft_s": {
        "mean": 75.144,
        "min": 41.759,
        "max": 100.609
      },
      "total_s": {
        "mean": 75.189,
        "min": 41.76,
        "max": 100.741
      },
      "tokens_per_s": {
        "mean": 13332.4,
        "min": 1664.1,
        "max": 28730.6
      },
      "input_tokens": {
        "mean": 1290.333,
        "min": 1153,
        "max": 1403
      },
      "output_tokens": {
        "mean": 89.667,
        "min": 15,
        "max": 221
      }
    },
    "gemini-3-flash-preview": {
      "calls": 6,
      "ok": 6,
      "failed": 0,
      "errors": [],
      "ttft_s": {
        "mean": 7.366,
        "min": 2.886,
        "max": 14.797
      },
      "total_s": {
        "mean": 7.737,
        "min": 2.886,
        "max": 15.43
      },
      "tokens_per_s": {
        "mean": 15727.517,
        "min": 211.0,
        "max": 55353.7
      },
      "input_tokens": {
        "mean": 1290.333,
        "min": 1153,
        "max": 1403
      },
      "output_tokens": {
        "mean": 125.667,
        "min": 32,
        "max": 183
      }
    },
    "gemini-3.1-flash-lite": {
      "calls": 6,
      "ok": 6,
      "failed": 0,
      "errors": [],
      "ttft_s": {
        "mean": 3.369,
        "min": 1.027,
        "max": 7.825
      },
      "total_s": {
        "mean": 3.472,
        "min": 1.028,
        "max": 8.132
      },
      "tokens_per_s": {
        "mean": 9860.7,
        "min": 520.8,
        "max": 40952.1
      },
      "input_tokens": {
        "mean": 1290.333,
        "min": 1153,
        "max": 1403
      },
      "output_tokens": {
        "mean": 90.167,
        "min": 32,
        "max": 160
      }
    }
  }
}
```

## ✅ 2026-08-25 19:15 — benchmark embedding: octen-0.6b

- **ทดลองอะไร:** วัดความเร็วและคุณภาพของ bflhc/Octen-Embedding-0.6B (596M, มิติ 1024)
- **ทำอย่างไร:** เข้ารหัส 3128 chunks บน cuda, batch=32, max_seq=768; ประเมิน dense-only ด้วย golden set 128 คำถาม
- **ผลลัพธ์:** โหลด 17.82s, เข้ารหัส 78.07s (40.06 chunks/s), ต่อคำถาม p50 87.63ms, Recall@1 47.7%, MRR 0.519

```json
{
  "model_key": "octen-0.6b",
  "hf_id": "bflhc/Octen-Embedding-0.6B",
  "params_m": 596,
  "dim": 1024,
  "max_seq_length": 768,
  "device": "cuda",
  "batch_size": 32,
  "n_chunks": 3128,
  "n_queries": 128,
  "load_seconds": 17.82,
  "encode_seconds": 78.07,
  "chunks_per_second": 40.06,
  "index_seconds": 0.091,
  "query_ms_mean": 87.96,
  "query_ms_p50": 87.63,
  "query_ms_p95": 116.92,
  "embeddings_mb": 12.2,
  "rss_before_gb": 0.5,
  "rss_after_gb": 1.84,
  "measured_at": "2026-08-25T19:15:58",
  "quality": {
    "all": {
      "mrr": 0.5191437251984127,
      "hit@1": 0.4765625,
      "recall@1": 0.4765625,
      "ndcg@1": 0.4765625,
      "hit@3": 0.5390625,
      "recall@3": 0.5390625,
      "ndcg@3": 0.511904054799108,
      "hit@5": 0.5546875,
      "recall@5": 0.5546875,
      "ndcg@5": 0.5182910029655763,
      "hit@10": 0.65625,
      "recall@10": 0.65625,
      "ndcg@10": 0.5506151667058069
    },
    "by_name": {
      "mrr": 0.9423363095238095,
      "hit@1": 0.90625,
      "recall@1": 0.90625,
      "ndcg@1": 0.90625,
      "hit@3": 0.984375,
      "recall@3": 0.984375,
      "ndcg@3": 0.9514498321986621,
      "hit@5": 0.984375,
      "recall@5": 0.984375,
      "ndcg@5": 0.9514498321986621,
      "hit@10": 1.0,
      "recall@10": 1.0,
      "ndcg@10": 0.9566581655319955
    },
    "by_code": {
      "mrr": 0.09595114087301587,
      "hit@1": 0.046875,
      "recall@1": 0.046875,
      "ndcg@1": 0.046875,
      "hit@3": 0.09375,
      "recall@3": 0.09375,
      "ndcg@3": 0.07235827739955403,
      "hit@5": 0.125,
      "recall@5": 0.125,
      "ndcg@5": 0.0851321737324905,
      "hit@10": 0.3125,
      "recall@10": 0.3125,
      "ndcg@10": 0.1445721678796184
    }
  }
}
```

## ✅ 2026-08-25 19:18 — benchmark sparse retrievers (dense=bge-m3)

- **ทดลองอะไร:** เทียบ 6 วิธี sparse นอกเหนือจาก BM25 ทั้งแบบเดี่ยวและ fuse กับ dense
- **ทำอย่างไร:** golden set 128 คำถาม, dense index จาก bge-m3, RRF k=60, candidate_k=20
- **ผลลัพธ์:** sparse เดี่ยวดีสุด: dirichlet-lm Recall@1 34.4%; hybrid ดีสุด: dense+dirichlet-lm Recall@1 87.5%

```json
{
  "methods": [
    "bm25",
    "bm25l",
    "bm25plus",
    "tfidf-word",
    "tfidf-char",
    "dirichlet-lm"
  ],
  "build_seconds": {
    "bm25": 1.89,
    "bm25l": 1.72,
    "bm25plus": 1.36,
    "tfidf-word": 1.53,
    "tfidf-char": 2.67,
    "dirichlet-lm": 1.53
  },
  "sparse_recall_at_1": {
    "bm25": 0.3281,
    "bm25l": 0.1016,
    "bm25plus": 0.3281,
    "tfidf-word": 0.0938,
    "tfidf-char": 0.0234,
    "dirichlet-lm": 0.3438
  },
  "hybrid_recall_at_1": {
    "bm25": 0.7734,
    "bm25l": 0.4766,
    "bm25plus": 0.7812,
    "tfidf-word": 0.6875,
    "tfidf-char": 0.4531,
    "dirichlet-lm": 0.875
  },
  "hybrid_pinned_recall_at_1": {
    "bm25": 0.9766,
    "bm25l": 0.8906,
    "bm25plus": 0.9844,
    "tfidf-word": 0.8828,
    "tfidf-char": 0.8359,
    "dirichlet-lm": 0.9688
  }
}
```

## ✅ 2026-08-25 19:20 — benchmark sparse retrievers (dense=bge-m3)

- **ทดลองอะไร:** เทียบ 6 วิธี sparse นอกเหนือจาก BM25 ทั้งแบบเดี่ยวและ fuse กับ dense
- **ทำอย่างไร:** golden set 128 คำถาม, dense index จาก bge-m3, RRF k=60, candidate_k=20
- **ผลลัพธ์:** sparse เดี่ยวดีสุด: dirichlet-lm Recall@1 34.4%; hybrid ดีสุด: dense+dirichlet-lm Recall@1 87.5%

```json
{
  "methods": [
    "bm25",
    "bm25l",
    "bm25plus",
    "tfidf-word",
    "tfidf-char",
    "dirichlet-lm"
  ],
  "build_seconds": {
    "bm25": 2.05,
    "bm25l": 1.58,
    "bm25plus": 1.37,
    "tfidf-word": 1.45,
    "tfidf-char": 2.76,
    "dirichlet-lm": 1.3
  },
  "sparse_recall_at_1": {
    "bm25": 0.3281,
    "bm25l": 0.1016,
    "bm25plus": 0.3281,
    "tfidf-word": 0.0938,
    "tfidf-char": 0.0391,
    "dirichlet-lm": 0.3438
  },
  "hybrid_recall_at_1": {
    "bm25": 0.7734,
    "bm25l": 0.4766,
    "bm25plus": 0.7812,
    "tfidf-word": 0.6875,
    "tfidf-char": 0.5,
    "dirichlet-lm": 0.875
  },
  "hybrid_pinned_recall_at_1": {
    "bm25": 0.9766,
    "bm25l": 0.8906,
    "bm25plus": 0.9844,
    "tfidf-word": 0.8828,
    "tfidf-char": 0.875,
    "dirichlet-lm": 0.9688
  }
}
```

## ✅ 2026-08-25 19:24 — benchmark LLM API latency

- **ทดลองอะไร:** วัดเวลาตอบกลับของโมเดลภาษาแต่ละตัว (TTFT, เวลารวม, tokens/วินาที)
- **ทำอย่างไร:** 3 คำถามจริง × 2 ครั้ง/โมเดล, prompt สร้างจาก hybrid retrieval บน index ของ bge-m3, เรียกแบบ streaming ไม่มี retry
- **ผลลัพธ์:** gemini-3.5-flash: TTFT 3.862s รวม 4.133s, gemini-3.6-flash: TTFT 30.981s รวม 31.334s, gemini-3.7-flash: TTFT 79.455s รวม 79.456s, gemini-3-flash-preview: TTFT 7.258s รวม 7.591s, gemini-3.1-flash-lite: TTFT 1.11s รวม 1.214s

```json
{
  "providers_with_keys": {
    "gemini": true,
    "openai": false,
    "anthropic": false
  },
  "summary": {
    "gemini-3.5-flash": {
      "calls": 6,
      "ok": 6,
      "failed": 0,
      "errors": [],
      "ttft_s": {
        "mean": 3.862,
        "min": 2.459,
        "max": 6.301
      },
      "total_s": {
        "mean": 4.133,
        "min": 2.46,
        "max": 6.567
      },
      "tokens_per_s": {
        "mean": 32.083,
        "min": 11.9,
        "max": 65.0
      },
      "input_tokens": {
        "mean": 1290.333,
        "min": 1153,
        "max": 1403
      },
      "output_tokens": {
        "mean": 128.167,
        "min": 33,
        "max": 224
      }
    },
    "gemini-3.6-flash": {
      "calls": 6,
      "ok": 5,
      "failed": 1,
      "errors": [
        "ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
      ],
      "ttft_s": {
        "mean": 30.981,
        "min": 3.176,
        "max": 70.59
      },
      "total_s": {
        "mean": 31.334,
        "min": 3.178,
        "max": 70.853
      },
      "tokens_per_s": {
        "mean": 7.96,
        "min": 0.8,
        "max": 20.7
      },
      "input_tokens": {
        "mean": 1285.4,
        "min": 1153,
        "max": 1403
      },
      "output_tokens": {
        "mean": 120,
        "min": 32,
        "max": 224
      }
    },
    "gemini-3.7-flash": {
      "calls": 6,
      "ok": 1,
      "failed": 5,
      "errors": [
        "ReadTimeout: The read operation timed out",
        "ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
      ],
      "ttft_s": {
        "mean": 79.455,
        "min": 79.455,
        "max": 79.455
      },
      "total_s": {
        "mean": 79.456,
        "min": 79.456,
        "max": 79.456
      },
      "tokens_per_s": {
        "mean": 0.4,
        "min": 0.4,
        "max": 0.4
      },
      "input_tokens": {
        "mean": 1153,
        "min": 1153,
        "max": 1153
      },
      "output_tokens": {
        "mean": 32,
        "min": 32,
        "max": 32
      }
    },
    "gemini-3-flash-preview": {
      "calls": 6,
      "ok": 5,
      "failed": 1,
      "errors": [
        "ServerError: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.', 'status': 'UNAVAILABLE'}}"
      ],
      "ttft_s": {
        "mean": 7.258,
        "min": 2.789,
        "max": 14.598
      },
      "total_s": {
        "mean": 7.591,
        "min": 2.807,
        "max": 15.121
      },
      "tokens_per_s": {
        "mean": 17.7,
        "min": 10.7,
        "max": 32.0
      },
      "input_tokens": {
        "mean": 1285.4,
        "min": 1153,
        "max": 1403
      },
      "output_tokens": {
        "mean": 135.2,
        "min": 32,
        "max": 222
      }
    },
    "gemini-3.1-flash-lite": {
      "calls": 6,
      "ok": 6,
      "failed": 0,
      "errors": [],
      "ttft_s": {
        "mean": 1.11,
        "min": 0.971,
        "max": 1.227
      },
      "total_s": {
        "mean": 1.214,
        "min": 1.097,
        "max": 1.309
      },
      "tokens_per_s": {
        "mean": 72.15,
        "min": 28.9,
        "max": 123.3
      },
      "input_tokens": {
        "mean": 1290.333,
        "min": 1153,
        "max": 1403
      },
      "output_tokens": {
        "mean": 90,
        "min": 32,
        "max": 159
      }
    }
  }
}
```

## ✅ 2026-08-25 19:29 — pipeline steps 1-7: e5-base

- **ทดลองอะไร:** รัน pipeline ขั้น 1-7 ด้วย embedding model e5-base
- **ทำอย่างไร:** 9 ไฟล์จาก ['curriculum'], chunk_size=150, overlap=30, max_seq=512, device=cuda
- **ผลลัพธ์:** รวม 74.4s, ครบทุกขั้น

```json
{
  "model": "e5-base",
  "hf_id": "intfloat/multilingual-e5-base",
  "step_seconds": {
    "1/7 แยกข้อความจาก PDF": 6.7,
    "2/7 ตัด chunk": 9.39,
    "3/7 สร้าง embeddings": 27.85,
    "4/7 สร้าง FAISS index": 0.51,
    "5/7 แปลงคำถามเป็นเวกเตอร์": 9.77,
    "6/7 ค้นหาด้วย similarity": 8.95,
    "7/7 retrieval ครบระบบ": 9.55
  },
  "total_seconds": 74.36,
  "failed_steps": []
}
```

## ✅ 2026-08-25 19:30 — pipeline steps 1-2: bge-m3

- **ทดลองอะไร:** รัน pipeline ขั้น 1-2 ด้วย embedding model bge-m3
- **ทำอย่างไร:** 9 ไฟล์จาก ['curriculum'], chunk_size=150, overlap=30, max_seq=768, device=auto
- **ผลลัพธ์:** รวม 16.4s, ครบทุกขั้น

```json
{
  "model": "bge-m3",
  "hf_id": "BAAI/bge-m3",
  "step_seconds": {
    "1/7 แยกข้อความจาก PDF": 5.26,
    "2/7 ตัด chunk": 9.5
  },
  "total_seconds": 16.35,
  "failed_steps": []
}
```

## ✅ 2026-08-25 19:31 — pipeline steps 1-2: bge-m3

- **ทดลองอะไร:** รัน pipeline ขั้น 1-2 ด้วย embedding model bge-m3
- **ทำอย่างไร:** 9 ไฟล์จาก ['curriculum'], chunk_size=150, overlap=30, max_seq=768, device=auto
- **ผลลัพธ์:** รวม 16.0s, ครบทุกขั้น

```json
{
  "model": "bge-m3",
  "hf_id": "BAAI/bge-m3",
  "step_seconds": {
    "1/7 แยกข้อความจาก PDF": 5.28,
    "2/7 ตัด chunk": 9.16
  },
  "total_seconds": 16.0,
  "failed_steps": []
}
```

## ✅ 2026-08-25 19:32 — สรุปผลการทดลองรอบ 25 ส.ค. 2026

- **ทดลองอะไร:** เปลี่ยน embedding model ตาม Thai-MTEB leaderboard, เทียบความเร็ว/คุณภาพ, หาทางเลือกแทน BM25, วัดความเร็ว API และรัน pipeline 1-7 ใหม่กับข้อมูลที่เพิ่มเข้ามา
- **ทำอย่างไร:** สร้าง registry โมเดล + ระบบสลับตอนรัน, golden set 128 คำถามจากตาราง CLO, benchmark ทีละโมเดลบน RTX 5050 ผ่าน .venv-gpu, เทียบ sparse 6 วิธีบน dense เดียวกัน, วัด Gemini 5 โมเดลแบบ streaming
- **ผลลัพธ์:** embedding ดีสุด multilingual-e5-base (Recall@1 72.7%, เร็วกว่า bge-m3 3.2 เท่า); sparse ดีสุดเมื่อ fuse คือ Dirichlet-LM (87.5% เทียบ BM25 77.3%) แต่เมื่อปักหมุดรหัสวิชาแล้ว BM25+ ชนะ (98.4%); LLM เร็วสุด gemini-3.1-flash-lite (TTFT 1.11s เทียบค่าเริ่มต้น 3.86s); ข้อมูลโตจาก 1,782 เป็น 3,128 chunks

```json
{
  "models_tested": [
    "e5-base",
    "bge-m3",
    "pixie-rune",
    "octen-0.6b"
  ],
  "models_blocked": {
    "embeddinggemma-300m": "gated ต้องมี HF token",
    "nemotron-8b/kalm-12b/linq/sfr": "เกิน RAM 32GB"
  },
  "best_embedding": "e5-base",
  "best_sparse_fused": "dirichlet-lm",
  "best_full_stack": "dense+bm25plus+pin",
  "bugs_found": [
    "huggingface_hub retry ใช้ httpx client ที่ปิดแล้ว",
    "tok/s คำนวณผิดเมื่อ response มา chunk เดียว",
    "tfidf-char เตรียม query กับ document คนละแบบ",
    "PDF 2 ไฟล์เป็นสแกน"
  ],
  "open_items": [
    "ลำดับ LLM_FALLBACK_MODELS เรียงกลับด้านกับผลวัด",
    "ยังไม่ได้ลบ index เก่าที่กำพร้า ~16MB",
    "วัด reranker ใหม่บน GPU"
  ]
}
```

## ✅ 2026-08-25 20:14 — pipeline steps 1-7: bge-m3

- **ทดลองอะไร:** รัน pipeline ขั้น 1-7 ด้วย embedding model bge-m3
- **ทำอย่างไร:** 23 ไฟล์จาก ['curriculum'], chunk_size=150, overlap=30, max_seq=768, device=auto
- **ผลลัพธ์:** รวม 978.1s, ครบทุกขั้น

```json
{
  "model": "bge-m3",
  "hf_id": "BAAI/bge-m3",
  "step_seconds": {
    "1/7 แยกข้อความจาก PDF": 4.51,
    "2/7 ตัด chunk": 8.46,
    "3/7 สร้าง embeddings": 927.67,
    "4/7 สร้าง FAISS index": 0.68,
    "5/7 แปลงคำถามเป็นเวกเตอร์": 13.76,
    "6/7 ค้นหาด้วย similarity": 9.19,
    "7/7 retrieval ครบระบบ": 12.36
  },
  "total_seconds": 978.12,
  "failed_steps": []
}
```

## ✅ 2026-08-25 20:38 — benchmark embedding: e5-base

- **ทดลองอะไร:** วัดความเร็วและคุณภาพของ intfloat/multilingual-e5-base (278M, มิติ 768)
- **ทำอย่างไร:** เข้ารหัส 3237 chunks บน cuda, batch=32, max_seq=512; ประเมิน dense-only ด้วย golden set 128 คำถาม
- **ผลลัพธ์:** โหลด 9.34s, เข้ารหัส 17.47s (185.29 chunks/s), ต่อคำถาม p50 9.57ms, Recall@1 72.7%, MRR 0.803

```json
{
  "model_key": "e5-base",
  "hf_id": "intfloat/multilingual-e5-base",
  "params_m": 278,
  "dim": 768,
  "max_seq_length": 512,
  "device": "cuda",
  "batch_size": 32,
  "n_chunks": 3237,
  "n_queries": 128,
  "corpus": {
    "n_chunks": 3237,
    "n_sources": 23,
    "chunks_sha1": "558fa8e941670876",
    "key_version": 1
  },
  "load_seconds": 9.34,
  "encode_seconds": 17.47,
  "chunks_per_second": 185.29,
  "index_seconds": 0.031,
  "query_ms_mean": 10.1,
  "query_ms_p50": 9.57,
  "query_ms_p95": 12.34,
  "embeddings_mb": 9.5,
  "rss_before_gb": 0.5,
  "rss_after_gb": 1.75,
  "measured_at": "2026-08-25T20:38:02",
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

## ✅ 2026-08-25 20:39 — benchmark embedding: bge-m3

- **ทดลองอะไร:** วัดความเร็วและคุณภาพของ BAAI/bge-m3 (568M, มิติ 1024)
- **ทำอย่างไร:** เข้ารหัส 3237 chunks บน cuda, batch=32, max_seq=768; ประเมิน dense-only ด้วย golden set 128 คำถาม
- **ผลลัพธ์:** โหลด 9.79s, เข้ารหัส 51.73s (62.58 chunks/s), ต่อคำถาม p50 17.06ms, Recall@1 63.3%, MRR 0.707

```json
{
  "model_key": "bge-m3",
  "hf_id": "BAAI/bge-m3",
  "params_m": 568,
  "dim": 1024,
  "max_seq_length": 768,
  "device": "cuda",
  "batch_size": 32,
  "n_chunks": 3237,
  "n_queries": 128,
  "corpus": {
    "n_chunks": 3237,
    "n_sources": 23,
    "chunks_sha1": "558fa8e941670876",
    "key_version": 1
  },
  "load_seconds": 9.79,
  "encode_seconds": 51.73,
  "chunks_per_second": 62.58,
  "index_seconds": 0.035,
  "query_ms_mean": 18.41,
  "query_ms_p50": 17.06,
  "query_ms_p95": 22.24,
  "embeddings_mb": 12.6,
  "rss_before_gb": 0.5,
  "rss_after_gb": 1.73,
  "measured_at": "2026-08-25T20:39:31",
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

## ✅ 2026-08-25 20:41 — benchmark sparse retrievers (dense=e5-base)

- **ทดลองอะไร:** เทียบ 6 วิธี sparse นอกเหนือจาก BM25 ทั้งแบบเดี่ยวและ fuse กับ dense
- **ทำอย่างไร:** golden set 128 คำถาม, dense index จาก e5-base, RRF k=60, candidate_k=20
- **ผลลัพธ์:** sparse เดี่ยวดีสุด: bm25 Recall@1 33.6%; hybrid ดีสุด: dense+dirichlet-lm Recall@1 92.2%

```json
{
  "methods": [
    "bm25",
    "bm25l",
    "bm25plus",
    "tfidf-word",
    "tfidf-char",
    "dirichlet-lm"
  ],
  "build_seconds": {
    "bm25": 2.29,
    "bm25l": 1.94,
    "bm25plus": 1.95,
    "tfidf-word": 2.05,
    "tfidf-char": 3.26,
    "dirichlet-lm": 1.92
  },
  "sparse_recall_at_1": {
    "bm25": 0.3359,
    "bm25l": 0.125,
    "bm25plus": 0.3281,
    "tfidf-word": 0.1016,
    "tfidf-char": 0.0391,
    "dirichlet-lm": 0.2656
  },
  "hybrid_recall_at_1": {
    "bm25": 0.8906,
    "bm25l": 0.5,
    "bm25plus": 0.8984,
    "tfidf-word": 0.7969,
    "tfidf-char": 0.625,
    "dirichlet-lm": 0.9219
  },
  "hybrid_pinned_recall_at_1": {
    "bm25": 0.9688,
    "bm25l": 0.8672,
    "bm25plus": 0.9766,
    "tfidf-word": 0.9141,
    "tfidf-char": 0.9062,
    "dirichlet-lm": 0.9453
  }
}
```

## ❌ 2026-08-25 20:44 — บั๊ก: golden set ผูกกับตำแหน่ง chunk จึงเพี้ยนเงียบๆ ตอนเพิ่มข้อมูล

- **ทดลองอะไร:** ตรวจว่าตัวเลข Recall/MRR ที่บันทึกไว้ยังใช้เทียบกับ corpus ปัจจุบันได้หรือไม่
- **ทำอย่างไร:** ไล่ดูว่า relevant_chunk_ids ใน data/golden_set.json ชี้ไปที่ chunk ไหนใน outputs/chunks.json ตอนนี้
- **ผลลัพธ์:** chunk_id เป็นเลขตำแหน่ง พอเพิ่มไฟล์ทะเบียน 14 ไฟล์ที่เรียงมาก่อน CLOs ทุก chunk ถูกนับใหม่ เฉลย "0" ที่เคยเป็นการ์ดวิชา กลายเป็นหน้าปกของ 00_สารบัญ (ที่ถูกคือ 109) ทั้ง 128 คำถามชี้ผิดหมด ไม่มี error ฟ้อง เพราะ guard เดิมเช็คแค่ว่า id มีอยู่จริง ซึ่ง "0" มีอยู่เสมอ — eval จะอ่านได้ 0% เงียบๆ

```json
{
  "root_cause": "relevant_chunk_ids เป็นตำแหน่ง ไม่ใช่ identity",
  "detection_gap": "build_golden_set.py เช็คว่า id resolve ได้ แต่ไม่เช็คว่า resolve ไปถูกตัว",
  "fix": "evaluation/golden_set.py: relevant_chunk_keys = sha1(source+text)[:16] + loader ที่ผูก id ใหม่ทุกครั้ง"
}
```

## ✅ 2026-08-25 20:44 — ปรับ default ให้ตรงกับผลวัด: e5-base + bm25plus + เรียง LLM fallback ใหม่

- **ทดลองอะไร:** ค่า default ใน config หลายตัวขัดกับผลวัดของตัวเอง ค้างเป็น open item มาตั้งแต่รอบ 19:32
- **ทำอย่างไร:** ซ่อม golden set ก่อน แล้ววัด e5-base กับ bge-m3 ใหม่บน corpus ปัจจุบัน 3,237 chunks บน RTX 5050 ผ่าน .venv-gpu จากนั้นเทียบ sparse 6 วิธีบน dense ตัวที่ชนะ
- **ผลลัพธ์:** e5-base ยืนยันชนะบน corpus ใหม่ (Recall@1 72.7% เทียบ 63.3%, encode เร็วกว่า 3 เท่า) → DEFAULT_MODEL; bm25plus ชนะในคอลัมน์ที่ใช้จริงคือ hybrid+ปักหมุด (97.7% เทียบ bm25 96.9%, dirichlet-lm 94.5%) → SPARSE_METHOD; LLM fallback เรียงใหม่ตามความน่าเชื่อถือ เอา 3.1-flash-lite (ผ่าน 12/12, TTFT 1.1-3.4s) ขึ้นหัว และ 3.7-flash (ผ่าน 4/12, TTFT 75-79s) ลงท้าย

```json
{
  "changed": {
    "src/model_registry.py": "DEFAULT_MODEL: bge-m3 -> e5-base",
    "config.py": "SPARSE_METHOD: bm25 -> bm25plus; LLM_FALLBACK_MODELS เรียงใหม่",
    "src/generator.py": "default model: gemini-3.6-flash -> gemini-3.5-flash (ขัดกับ config)"
  },
  "embedding": {
    "e5-base": {
      "recall@1": 0.727,
      "encode_s": 17.5
    },
    "bge-m3": {
      "recall@1": 0.633,
      "encode_s": 51.7
    }
  },
  "sparse_hybrid_pinned": {
    "bm25plus": 0.977,
    "bm25": 0.969,
    "dirichlet-lm": 0.945,
    "tfidf-word": 0.914,
    "tfidf-char": 0.906,
    "bm25l": 0.867
  },
  "corpus": {
    "n_chunks": 3237,
    "n_sources": 23,
    "chunks_sha1": "558fa8e941670876"
  },
  "still_open": [
    "ยังไม่ลบ index กำพร้า ~38MB",
    "reranker ยังไม่วัดบน GPU",
    "golden set ยังครอบคลุมแค่ CLOs 64/3237 chunks",
    "PDF สแกน 2 ไฟล์ยังไม่ OCR"
  ]
}
```
