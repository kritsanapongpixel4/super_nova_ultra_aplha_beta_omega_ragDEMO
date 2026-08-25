"""Registry of every embedding model the system can be pointed at.

The point of this module is that changing embedding model is *data*, not a
code edit.  Pick one by key — on the command line (``--model qwen3-0.6b``),
through the ``RAG_EMBED_MODEL`` environment variable, or from the web chat's
dropdown — and config.py derives the dimension, the prompt format and a
separate set of index paths from the entry below.  Nothing has to be edited
in config.py to switch, and two models' indexes never overwrite each other.

Each entry records what the benchmark needs to know *before* downloading
9-24 GB of weights: how big it is, how much RAM it wants, and whether this
machine can realistically run it.  ``python -m src.model_registry`` prints
the table.

Prompt formats are not cosmetic.  E5 and Qwen3-style models were trained
with an asymmetric prefix on the query side; dropping it costs real recall,
and applying a *document* prefix to a query silently puts the two in
different regions of the space.  The strings below are copied from each
model's own ``config_sentence_transformers.json`` (checked 2026-08-25), not
guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

# Why 768 tokens and not the 8k-32k each model would allow: measured over
# this corpus's 3,128 chunks on 2026-08-25, the longest chunk is 271 tokens
# under the XLM-R tokenizer (bge-m3, e5, PIXIE) and 699 under Qwen3's, whose
# vocabulary splits Thai far more finely.  768 truncates nothing at all,
# while Qwen3's 32k default would reserve buffers for a context 45x longer
# than anything in the data.  Attention is quadratic — that is not free.

# The instruction Qwen3-Embedding and its derivatives were trained to see on
# the query side.  Written once — four models share it verbatim.
_QWEN3_QUERY = (
    "Instruct: Given a web search query, retrieve relevant passages that "
    "answer the query\nQuery:"
)


@dataclass(frozen=True)
class EmbeddingSpec:
    """Everything needed to load, prompt, size up and file away one model."""

    key: str                    # short name used on the CLI and in paths
    hf_id: str                  # HuggingFace repo id
    dim: int                    # output dimension → FAISS index width
    params_m: int               # parameter count in millions
    weights_gb: float           # download size, measured from the HF file tree
    query_prompt: str = ""      # prepended when embedding a question
    doc_prompt: str = ""        # prepended when embedding a chunk
    max_seq_length: int = 768   # tokens per chunk; measured, see MAX_SEQ note
    gated: bool = False         # needs an accepted licence + HF token
    tier: str = "small"         # small | large | infeasible
    note: str = ""

    @property
    def ram_fp32_gb(self) -> float:
        """Rough resident size once loaded in float32, plus activations.

        sentence-transformers loads in float32 on CPU regardless of the
        checkpoint's dtype, so a bf16 checkpoint still costs 4 bytes per
        parameter here — which is exactly the number that decides whether a
        model fits in 32 GB.
        """
        return round(self.params_m * 4 / 1024 + 1.5, 1)

    @property
    def slug(self) -> str:
        """Filesystem-safe name for this model's own index directory."""
        return self.key


# Ordered as they appear on the Thai-MTEB leaderboard, with the two models
# the project already had measurements for kept at the end as baselines.
MODELS: dict[str, EmbeddingSpec] = {
    spec.key: spec
    for spec in (
        # ── Runs comfortably on this machine ────────────────────────────
        EmbeddingSpec(
            key="e5-base",
            hf_id="intfloat/multilingual-e5-base",
            dim=768,
            params_m=278,
            weights_gb=1.11,
            # E5 is the one family here that ships no prompt config, so the
            # prefixes have to be applied by hand.  Its model card is
            # explicit that both sides need one and that they differ.
            query_prompt="query: ",
            doc_prompt="passage: ",
            max_seq_length=512,   # hard limit: 514 learned position embeddings
            tier="small",
            note="ฐานเทียบเดิมของโปรเจกต์ เร็วที่สุดในชุดนี้",
        ),
        EmbeddingSpec(
            key="embeddinggemma-300m",
            hf_id="google/embeddinggemma-300m",
            dim=768,
            params_m=308,
            weights_gb=1.20,
            query_prompt="task: search result | query: ",
            doc_prompt="title: none | text: ",
            max_seq_length=768,
            gated=True,
            tier="small",
            note="อันดับ 2 บน Thai-MTEB — ต้องยอมรับ licence + ใช้ HF token",
        ),
        EmbeddingSpec(
            key="pixie-rune",
            hf_id="telepix/PIXIE-Rune-v1.0",
            dim=1024,
            params_m=568,
            weights_gb=2.27,
            query_prompt="query: ",
            doc_prompt="",
            max_seq_length=768,
            tier="small",
            note="อันดับ 8 — สถาปัตยกรรมเดียวกับ bge-m3 (XLM-R large)",
        ),
        EmbeddingSpec(
            key="octen-0.6b",
            hf_id="bflhc/Octen-Embedding-0.6B",
            dim=1024,
            params_m=596,
            weights_gb=1.19,
            query_prompt=_QWEN3_QUERY,
            # Its config really does say a single space, not an empty string.
            doc_prompt=" ",
            max_seq_length=768,
            tier="small",
            note="อันดับ 6 — ต่อยอดจาก Qwen3-Embedding-0.6B",
        ),
        EmbeddingSpec(
            key="qwen3-0.6b",
            hf_id="Qwen/Qwen3-Embedding-0.6B",
            dim=1024,
            params_m=596,
            weights_gb=1.19,
            query_prompt=_QWEN3_QUERY,
            doc_prompt="",
            max_seq_length=768,
            tier="small",
            note="ไม่ได้อยู่ในรูป — ใส่เป็นตัวต้นทางของ Octen เพื่อแยกผลของการ fine-tune",
        ),
        EmbeddingSpec(
            key="bge-m3",
            hf_id="BAAI/bge-m3",
            dim=1024,
            params_m=568,
            weights_gb=2.27,
            max_seq_length=768,
            tier="small",
            note="ค่าเริ่มต้นปัจจุบันของระบบ",
        ),
        # ── Runs, but slowly ────────────────────────────────────────────
        EmbeddingSpec(
            key="qwen3-4b",
            hf_id="Qwen/Qwen3-Embedding-4B",
            dim=2560,
            params_m=4022,
            weights_gb=8.04,
            query_prompt=_QWEN3_QUERY,
            doc_prompt="",
            max_seq_length=768,
            tier="large",
            note="อันดับ 4 — 16 GB RAM ใน fp32, ไม่พอลง VRAM 8 GB แม้ใน fp16",
        ),
        EmbeddingSpec(
            key="boom-4b",
            hf_id="ICT-TIME-and-Querit/BOOM_4B_v1",
            dim=2560,
            params_m=4022,
            weights_gb=16.09,
            query_prompt=_QWEN3_QUERY,
            doc_prompt="",
            max_seq_length=768,
            tier="large",
            note="อันดับ 5 — น้ำหนักเก็บเป็น fp32 จึงดาวน์โหลด 16 GB",
        ),
        # ── Documented but out of reach on 32 GB RAM / 8 GB VRAM ────────
        EmbeddingSpec(
            key="linq-mistral",
            hf_id="Linq-AI-Research/Linq-Embed-Mistral",
            dim=4096,
            params_m=7111,
            weights_gb=14.22,
            query_prompt=_QWEN3_QUERY,
            tier="infeasible",
            note="อันดับ 7 — ต้องการ ~28 GB RAM ใน fp32",
        ),
        EmbeddingSpec(
            key="sfr-mistral",
            hf_id="Salesforce/SFR-Embedding-Mistral",
            dim=4096,
            params_m=7111,
            weights_gb=14.26,
            query_prompt=_QWEN3_QUERY,
            tier="infeasible",
            note="อันดับ 9 — ต้องการ ~28 GB RAM ใน fp32",
        ),
        EmbeddingSpec(
            key="nemotron-8b",
            hf_id="nvidia/llama-embed-nemotron-8b",
            dim=4096,
            params_m=7505,
            weights_gb=15.01,
            query_prompt=_QWEN3_QUERY,
            tier="infeasible",
            note="อันดับ 1 — ต้องการ ~30 GB RAM ใน fp32",
        ),
        EmbeddingSpec(
            key="kalm-gemma3-12b",
            hf_id="tencent/KaLM-Embedding-Gemma3-12B-2511",
            dim=3840,
            params_m=11766,
            weights_gb=23.53,
            query_prompt=_QWEN3_QUERY,
            tier="infeasible",
            note="อันดับ 3 — ต้องการ ~47 GB RAM ใน fp32, เกินเครื่องนี้",
        ),
    )
}

DEFAULT_MODEL = "bge-m3"

#: Keys the benchmark runs by default — everything the machine can finish.
RUNNABLE = [key for key, spec in MODELS.items() if spec.tier != "infeasible"]


def get(key: str) -> EmbeddingSpec:
    """Look up a model by key, or by its full HuggingFace id.

    Accepting the HF id too means an existing ``EMBEDDING_MODEL =
    "BAAI/bge-m3"`` in config.py keeps working unchanged.
    """
    if key in MODELS:
        return MODELS[key]
    for spec in MODELS.values():
        if spec.hf_id.lower() == key.lower():
            return spec
    raise KeyError(
        f"ไม่รู้จักโมเดล {key!r} — ที่มีให้เลือก: {', '.join(MODELS)}"
    )


def resolve(key: str) -> EmbeddingSpec:
    """Like get(), but a model outside the registry is allowed through.

    Someone trying a fourteenth model should not have to edit this file
    first.  The dimension is then unknown until the model is loaded, so it
    is left at 0 and filled in by EmbeddingModel once the weights are up.
    """
    try:
        return get(key)
    except KeyError:
        if "/" not in key:
            raise
        return EmbeddingSpec(
            key=key.replace("/", "__"),
            hf_id=key,
            dim=0,
            params_m=0,
            weights_gb=0.0,
            tier="unknown",
            note="ไม่ได้อยู่ใน registry — มิติจะถูกอ่านจากตัวโมเดลตอนโหลด",
        )


def table() -> str:
    """A printable feasibility table — the answer to 'will this run?'."""
    header = (
        f"{'key':20s} {'params':>8s} {'dim':>5s} {'ดาวน์โหลด':>10s} "
        f"{'RAM fp32':>9s}  tier"
    )
    lines = [header, "-" * len(header)]
    for spec in MODELS.values():
        lines.append(
            f"{spec.key:20s} {spec.params_m:>7,}M {spec.dim:>5d} "
            f"{spec.weights_gb:>9.2f}G {spec.ram_fp32_gb:>8.1f}G  {spec.tier}"
            + ("  [gated]" if spec.gated else "")
        )
    return "\n".join(lines)


if __name__ == "__main__":  # python -m src.model_registry
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    print(table())
