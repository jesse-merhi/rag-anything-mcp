# rag_anything_server.py  – single shared workspace
import shutil
import os, typer, asyncio, json
from pathlib import Path
from typing import List, Literal, Optional, Dict
import logging

from mcp.server.fastmcp import FastMCP, Context
from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from enum import Enum
mcp = FastMCP("rag-anything")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("rag_anything_server")

# Cross platform shared working directory, naturally if you have specified your own in the config, then it will be used.
default_working_dir = Path.home() / ".rag_anything" / "shared_workspace"
SHARED_WORKDIR = os.environ.get("RAG_ANYTHING_WORKING_DIR", str(default_working_dir))
# Ensure the working directory exists
SHARED_WORKDIR = os.path.expanduser(SHARED_WORKDIR)
Path(SHARED_WORKDIR).mkdir(parents=True, exist_ok=True)

# Cross platform output directory
default_output_dir = Path.home() / ".rag_anything" / "output"
OUTPUT_DIR = os.environ.get("RAG_ANYTHING_OUTPUT_DIR", str(default_output_dir))
# Ensure the output directory exists
OUTPUT_DIR = os.path.expanduser(OUTPUT_DIR)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# OpenAI API key pulled in from here
_api_key = os.getenv("OPENAI_API_KEY", "")
# Ensure the API key is set
if not _api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

_global_rag: RAGAnything | None = None

class QueryMode(Enum):
    LOCAL = "local"
    GLOBAL = "global"
    HYBRID = "hybrid"
    NAIVE = "naive"
    MIX = "mix"
    BYPASS = "bypass"

class MinerUParseMethod(Enum):
    AUTO = "auto"
    OCR = "ocr"
    TEXT = "txt"

def _llm(api_key: str):
    """
    Create a function to call the LLM with caching. 
    Can set the RAG_ANYTHING_LLM_MODEL environment variable to change the model.
    Is likely only supported for OpenAI models. Until lightrag supports more models.
    """
    model = os.environ.get("RAG_ANYTHING_LLM_MODEL", "gpt-4o-mini")
    logging.info(f"Creating llm function with model: {model}")
    return lambda p, system_prompt=None, history_messages=[], **kw: \
        openai_complete_if_cache(
            model, p,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            **kw,
        )


def _vision(api_key: str):
    """
    Returns a function that can handle multimodal inputs with images.
    Uses the OpenAI API to process images if image_data is provided.
    Otherwise, it falls back to the standard LLM function.
    This is useful for multimodal queries where images are involved.
    """
    def fn(prompt, system_prompt=None, history_messages=[], image_data=None, **kw):
        model = os.environ.get("RAG_ANYTHING_IMAGE_MODEL", "gpt-4.1")
        image_processing_prompt = os.environ.get("RAG_ANYTHING_IMAGE_PROCESSING_PROMPT","")
        if image_data:
            logging.info(f"Creating vision function with model: {model} and prompt: {image_processing_prompt}")
            return openai_complete_if_cache(
                model, image_processing_prompt,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}]}],
                api_key=api_key, **kw)
        return _llm(api_key)(prompt, system_prompt, history_messages, **kw)
    return fn


def _embed(api_key: str):
    model = os.environ.get("RAG_ANYTHING_EMBEDDING_MODEL", "text-embedding-3-large")
    logging.info(f"Creating embedding function with model: {model}")
    return EmbeddingFunc(
        embedding_dim=3072,
        max_token_size=8192,
        func=lambda t: openai_embed(
            t, model=model, api_key=api_key),
    )


async def _get_rag() -> RAGAnything:
    """Create the shared RAGAnything, dont create it if it already exists."""
    global _global_rag
    if _global_rag:
        return _global_rag
    lr = LightRAG(
        working_dir=SHARED_WORKDIR,
        llm_model_func=_llm(_api_key),
        embedding_func=_embed(_api_key),
    )
    await lr.initialize_storages()
    await initialize_pipeline_status()
    logging.info(f"Creating new rag on {SHARED_WORKDIR}")
    _global_rag = RAGAnything(
        lightrag=lr,
        llm_model_func=lr.llm_model_func,
        vision_model_func=_vision(_api_key),
        embedding_func=lr.embedding_func,
        config=RAGAnythingConfig(
            working_dir=SHARED_WORKDIR,
            mineru_parse_method="auto",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        ),
    )
    return _global_rag


async def _already_ingested(lr: LightRAG, abs_file: str, ctx: Context | None = None) -> bool:
    """
    Function to check if a document has already been processed by LightRAG. 
    """
    logging.info(f"Checking if {abs_file} has been ingested.")
    processed = await lr.doc_status.get_docs_by_status(DocStatus.PROCESSED)
    if ctx:
        await ctx.debug(f"processed ids: {list(processed.values())}")
    return any(Path(abs_file).name.lower() == s.file_path.lower()
               for s in processed.values())


async def _ensure_doc(rag: RAGAnything, file_path: str, parse_method: MinerUParseMethod, ctx: Context | None = None): ## TODO: We need to type the parse method so it doesnt fk it up.
    """
    Checks if the document is already ingested, and if not, processes it.
    """
    abs_fp = os.path.abspath(file_path)
    if await _already_ingested(rag.lightrag, abs_fp, ctx):
        return
    
    logging.info(f"Processing {file_path} has been ingested.")
    await rag.process_document_complete(
        file_path=abs_fp, output_dir=OUTPUT_DIR,
        parse_method=parse_method, device="cuda:0", lang="en")


@mcp.tool()
async def process_directory(
    directory_path: str,
    file_extensions: Optional[List[str]] = None,
    recursive: bool = True,
    max_workers: int = 4,
):
    """
    process_directory
    • If every matching file is already ingested → return success immediately.
    • Otherwise call rag.process_folder_complete for fast, threaded ingestion.
    """
    if not os.path.isdir(directory_path):
        return f"Error: '{directory_path}' is not a directory."

    rag = await _get_rag()
    exts = file_extensions or [
        ".pdf", ".docx", ".pptx", ".txt", ".md", ".ppt", ".rtf"
    ]

    # Collect candidate files
    candidate_files = [
        p for p in Path(directory_path).rglob("*" if recursive else "*.*")
        if p.is_file() and p.suffix.lower() in exts
    ]


    logging.info(f"Files to ingest: {candidate_files}")
    # Filter out the ones already ingested
    to_process: list[Path] = []
    for fp in candidate_files:
        if not await _already_ingested(rag.lightrag, str(fp)):
            to_process.append(fp)

    if not to_process:
        return (
            f"✅ All {len(candidate_files)} matching file(s) in "
            f"'{directory_path}' are already indexed."
        )

    # New/changed docs exist → process entire folder in parallel
    logging.info(f"Processing folder: {directory_path}")
    await rag.process_folder_complete(
        folder_path=directory_path,
        output_dir=OUTPUT_DIR,
        file_extensions=exts,
        recursive=recursive,
        max_workers=max_workers,
    )
    return (
        f"Indexed {len(to_process)} new or updated file(s) into "
        f"shared workspace '{SHARED_WORKDIR}'."
    )


@mcp.tool()
async def process_single_document(file_path: str | None = None, parse_method: MinerUParseMethod = "auto"):
    """ 
    process_single_document
    Processes a single document, indexing it into the shared workspace.
    If `file_path` is not provided, it returns an error message.
    If the file does not exist, it returns an error message.
    The `parse_method` can be specified to control how the document is processed.
    Supported parse methods include "auto", "pdfminer", "pypdf", "docx", "pptx", etc.
    """
    if not file_path:
        return "Error: please pass file_path."
    if not os.path.isfile(file_path):
        return f"Error: '{file_path}' is not a file."

    rag = await _get_rag()
    logging.info(f"Ensuring {file_path} has been ingested.")
    await _ensure_doc(rag, file_path, parse_method)
    return f"✅ '{file_path}' indexed."


@mcp.tool()
async def check_doc_ingested(file_path: str, ctx: Context):
    """ 
    check_doc_ingested
    Checks if a document has already been processed by LightRAG.
    If `file_path` is not provided, it returns an error message.
    If the file does not exist, it returns an error message.
    """
    if not os.path.isfile(file_path):
        return f"Error: '{file_path}' is not a file."
    rag = await _get_rag()
    logging.info(f"Checking if {file_path} has been ingested.")
    if await _already_ingested(rag.lightrag, file_path, ctx):
        return f"✅ already indexed."
    return f"⚠ not indexed yet."


@mcp.tool()
async def query_workspace(query: str, mode: QueryMode = "hybrid"): 
    """ 
    query_workspace
    Sends a natural language query to obtain information about the indexed documents in the shared workspace.
    You need to have indexed documents first using `process_directory` or `process_single_document` for this 
    query to return meaningful results.

    The `mode` can be "hybrid", "retrieval", or "generation" to control how the query is processed.
    """
    rag = await _get_rag()
    logging.info(f"Executing query: `{query}`")
    return await rag.aquery(query, mode=mode)


@mcp.tool()
async def query_with_multimodal(query: str, multimodal_content: List[dict], mode: QueryMode = "hybrid"):
    """ 
    query_with_multimodal
    Sends a natural language query with multimodal content (like images) to obtain information about the indexed 
    documents in the shared workspace. You need to have indexed documents first using `process_directory` or 
    `process_single_document` for this query to return meaningful results.

    The `mode` can be "hybrid", "retrieval", or "generation" to control how the query is processed.
    """
    rag = await _get_rag()
    logging.info(f"Executing multimodal query: `{query}`")
    return await rag.aquery_with_multimodal(
        query, multimodal_content, mode=mode)


@mcp.tool()
async def get_workspace_info():
    """ get_workspace_info
    Returns information about the shared workspace, including the working directory and output directory.
    """
    rag = await _get_rag()
    logging.info(f"Getting Workspace Info")
    return rag.get_config_info()

@mcp.tool()
async def clear_all_data(confirm: bool = False):
    """
    clear_all_data
    ⚠ PERMANENTLY deletes everything in SHARED_WORKDIR *and* OUTPUT_DIR.

    Pass `confirm=True` (or `--yes` on the CLI) to actually perform the wipe.
    """
    if not confirm:
        return "Refused: pass confirm=True (CLI: --yes) to delete all stored data."

    global _global_rag
    _global_rag = None  # drop the in-memory RAG so a fresh one is built later

    for p in (SHARED_WORKDIR, OUTPUT_DIR):
        path = Path(p)               # ← ensure we have a Path object
        if path.exists():
            logger.warning("Deleting %s …", path)
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    logger.info("All data cleared; empty directories recreated.")
    return "🧹 Workspace and output directories have been wiped."

app = typer.Typer(
    help="RAG-Anything utility: either start the MCP server "
         "or use the same features from your shell.",
    no_args_is_help=True
)

@app.command()
def run():
    """Launch as an MCP server (stdio transport)."""
    mcp.run(transport="stdio")

cli = typer.Typer(help="CLI commands (same as the MCP tools)")
app.add_typer(cli, name="cli")

def _run(coro): return asyncio.run(coro)

@cli.command("process-directory")
def cli_process_directory(
    directory_path: str,
    file_extensions: Optional[List[str]] = typer.Option(
        None, "--ext", "-e", help="Filter by extensions, e.g. -e .pdf .md"),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", "-r/-R")
):
    typer.echo(_run(process_directory(directory_path, file_extensions, recursive)))

@cli.command("process-single-document")
def cli_process_single_document(
    file_path: str,
    parse_method: str = typer.Option("auto", "--parse-method", "-p")
):
    typer.echo(_run(process_single_document(file_path, parse_method)))

@cli.command("check-doc")
def cli_check_doc(file_path: str):
    typer.echo(_run(check_doc_ingested(file_path, None)))

@cli.command("query")
def cli_query(query: str, mode: str = typer.Option("hybrid", "--mode", "-m")):
    typer.echo(_run(query_workspace(query, mode)))

@cli.command("query-mm")
def cli_query_mm(
    query: str,
    multimodal_json: str = typer.Option(
        ..., "--content", "-c",
        help='JSON string or "@/path/to/file.json" with multimodal content'),
    mode: str = typer.Option("hybrid", "--mode", "-m")
):
    if multimodal_json.startswith("@"):
        multimodal_content = json.load(open(multimodal_json[1:], "r"))
    else:
        multimodal_content = json.loads(multimodal_json)
    typer.echo(_run(query_with_multimodal(query, multimodal_content, mode)))

@cli.command("workspace-info")
def cli_workspace_info():
    typer.echo(_run(get_workspace_info()))

@cli.command("clear-all-data")
def cli_clear_all_data(
    really: bool = typer.Option(
        False, "--yes", help="Actually delete everything (required).")
):
    typer.echo(_run(clear_all_data(confirm=really)))

if __name__ == "__main__":
    app() 