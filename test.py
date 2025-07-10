import asyncio, os, time
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from lightrag.base import DocStatus            # ← enum with “processed”, “failed”…

# ── parameters you’ll tweak ----------------------------------------------------
API_KEY   = os.getenv("OPENAI_API_KEY", "")
WORK_DIR  = "./rag_storage"
PDF_PATH  = "files/paper.pdf"
PARSE_MTD = "auto"           # "ocr", "txt", or "auto"
DEVICE    = "cuda:0"
# -----------------------------------------------------------------------------


# ---------- tiny timing helpers ----------------------------------------------
t0_glob = time.perf_counter()
def elapsed() -> str:
    return f"{(time.perf_counter()-t0_glob)*1000:,.0f} ms"
# -----------------------------------------------------------------------------


async def build_light_rag() -> LightRAG:
    lr = LightRAG(
        working_dir=WORK_DIR,
        llm_model_func=lambda p, system_prompt=None, history_messages=[], **kw:
            openai_complete_if_cache(
                "gpt-4o-mini", p, system_prompt=system_prompt,
                history_messages=history_messages, api_key=API_KEY, **kw),
        embedding_func=EmbeddingFunc(
            embedding_dim=3072, max_token_size=8192,
            func=lambda t: openai_embed(
                t, model="text-embedding-3-large", api_key=API_KEY)),
    )
    await lr.initialize_storages()
    await initialize_pipeline_status()
    print(f"🔸 LightRAG ready ({elapsed()})")
    return lr


async def pdf_already_ingested(lr: LightRAG, file_path: str) -> bool:
    """Return True iff doc-status store already shows this file as PROCESSED."""
    processed = await lr.doc_status.get_docs_by_status(DocStatus.PROCESSED)
    abs_target = str(Path(file_path).resolve())
    print(processed)
    return any(d.file_path == abs_target for d in processed.values())


async def ensure_document(rag_any, file_path: str,
                          parse_method: str = "auto", device: str = "cpu"):
    """
    Skip MinerU/LLM work if *any* processed DocStatus already lists
    the same file-name (case-insensitive).
    """
    # 1) ask the store for all PROCESSED docs  (works in JsonDocStatusStorage)
    processed = await rag_any.lightrag.doc_status.get_docs_by_status(
        DocStatus.PROCESSED
    )           # -> dict {doc_id: DocProcessingStatus}

    target_name = Path(file_path).name.lower()

    # 2) look for a matching file-name
    for ds in processed.values():
        if Path(ds.file_path).name.lower() == target_name:
            print(f"✅ '{file_path}' already indexed – skipping parse")
            return

    # 3) not found → run the full pipeline once
    print("📄 parsing new PDF …")
    await rag_any.process_document_complete(
        file_path=file_path,
        output_dir="./output",
        parse_method=parse_method,
        device=device,
        lang="en",
    )
    print("✔ parse finished")

async def main():
    lr  = await build_light_rag()

    ra = RAGAnything(
        lightrag=lr,
        llm_model_func=lr.llm_model_func,
        vision_model_func=lr.llm_model_func,    # reuse same OpenAI func
        embedding_func=lr.embedding_func,
        config=RAGAnythingConfig(
            working_dir=WORK_DIR,
            mineru_parse_method=PARSE_MTD,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        ),
    )

    await ensure_document(ra, PDF_PATH)

    # ----- sample queries -----------------------------------------------------
    tq = time.perf_counter()
    resp = await ra.aquery(
        "What are the main findings shown in the figures and tables?",
        mode="hybrid",
    )
    print(f"\n🗒  Text-only query in {(time.perf_counter()-tq)*1000:,.0f} ms")
    print(resp, "\n")

    tq = time.perf_counter()
    resp = await ra.aquery_with_multimodal(
        "Explain this formula and its relevance to the document content",
        multimodal_content=[{
            "type": "equation",
            "latex": r"P(d|q) = \frac{P(q|d)\,P(d)}{P(q)}",
            "equation_caption": "Document relevance probability",
        }],
        mode="hybrid",
    )
    print(f"🗒  Multimodal query   in {(time.perf_counter()-tq)*1000:,.0f} ms")
    print(resp)

if __name__ == "__main__":
    asyncio.run(main())
