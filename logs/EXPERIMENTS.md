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
