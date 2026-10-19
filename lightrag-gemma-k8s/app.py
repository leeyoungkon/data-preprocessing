"""LightRAG + Ollama local lab. CLI: python app.py --help"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete
from lightrag.utils import EmbeddingFunc
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("LLM_MODEL", "gemma3:12b")
EMBED = os.getenv("EMBED_MODEL", "bge-m3")
CTX = int(os.getenv("NUM_CTX", "16384"))
TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "900"))
STORAGE = Path(os.getenv("WORKING_DIR", "./rag_storage")).expanduser()
if not STORAGE.is_absolute():
    STORAGE = ROOT / STORAGE
MODES = {"naive", "local", "global", "hybrid", "mix"}
RULES = ("한국어로 답하세요. 검색된 문서에 있는 사실만 사용하세요. "
         "관계가 여러 단계이면 연결 경로를 설명하세요. 문서에 없는 전화번호, "
         "날짜, 금액은 추측하지 말고 확인할 수 없다고 하세요. 출처가 제공되면 문서명을 표시하세요.")


async def embed(texts: list[str]) -> np.ndarray:
    # Gemma is used for generation/extraction; bge-m3 is used only for embeddings.
    async with httpx.AsyncClient(timeout=TIMEOUT, trust_env=False) as client:
        response = await client.post(f"{HOST}/api/embed", json={
            "model": EMBED, "input": texts, "truncate": False,
            "keep_alive": "30m",
        })
        response.raise_for_status()
        vectors = np.asarray(response.json()["embeddings"], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise RuntimeError("Ollama가 올바른 임베딩 배열을 반환하지 않았습니다.")
    return vectors


async def check_ollama() -> int:
    try:
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            response = await client.get(f"{HOST}/api/tags")
            response.raise_for_status()
            available = {m["name"] for m in response.json().get("models", [])}
        for name in (MODEL, EMBED):
            resolved = name if ":" in name else f"{name}:latest"
            if name not in available and resolved not in available:
                raise RuntimeError(f"모델이 없습니다: {name}. Ollama 서버에서 ollama pull {name} 실행 필요.")
        return int((await embed(["연결 확인용 한국어 문장"])).shape[1])
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Ollama 연결/임베딩 실패: {HOST}. 서버 주소와 모델을 확인하세요. {exc}") from exc


async def create_rag() -> LightRAG:
    if CTX < 16384:
        raise RuntimeError("이 실습은 NUM_CTX=16384 이상을 사용합니다.")
    dim = await check_ollama()
    STORAGE.mkdir(parents=True, exist_ok=True)
    config_file = STORAGE / "lab_config.json"
    config = {"llm_model": MODEL, "embedding_model": EMBED, "embedding_dim": dim,
              "chunk_token_size": 500, "chunk_overlap_token_size": 60}
    if config_file.exists():
        if json.loads(config_file.read_text(encoding="utf-8")) != config:
            raise RuntimeError("기존 색인과 모델/청킹 설정이 다릅니다. .env의 WORKING_DIR을 새 폴더로 바꾸고 ingest 하세요.")
    else:
        config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    rag = LightRAG(
        working_dir=str(STORAGE),
        llm_model_func=ollama_model_complete,
        llm_model_name=MODEL,
        llm_model_max_async=1,
        default_llm_timeout=int(TIMEOUT),
        default_embedding_timeout=int(TIMEOUT),
        max_parallel_insert=1,
        llm_model_kwargs={"host": HOST, "timeout": TIMEOUT,
                          "options": {"num_ctx": CTX, "temperature": 0, "num_predict": 4096},
                          "keep_alive": "30m"},
        embedding_func=EmbeddingFunc(embedding_dim=dim, max_token_size=2048,
                                     model_name=EMBED, func=embed),
        embedding_func_max_async=1,
        chunk_token_size=500,
        chunk_overlap_token_size=60,
        entity_extract_max_gleaning=1,
        entity_extraction_use_json=False,
        summary_max_tokens=1024,
        addon_params={"language": "Korean"},
    )
    await rag.initialize_storages()
    return rag


async def ingest(rag: LightRAG) -> dict:
    files = sorted((ROOT / "docs").glob("*.txt"))
    if not files:
        raise RuntimeError("docs 폴더에 UTF-8 .txt 문서를 넣으세요.")
    manifest_file = STORAGE / "lab_documents.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists() else {}
    texts = {p.name: p.read_text(encoding="utf-8-sig") for p in files}
    hashes = {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in texts.items()}
    for name, digest in hashes.items():
        if name in manifest and manifest[name] != digest:
            raise RuntimeError(f"기존 문서 내용이 변경됐습니다: {name}. 새 WORKING_DIR에서 다시 ingest 하세요.")
    # Avoid LightRAG creating duplicate document failure records on repeated ingestion.
    # Persist each successful file so a later failure does not re-submit earlier files.
    for name, text in texts.items():
        if name in manifest or hashes[name] in manifest.values():
            print(f"건너뜀 (이미 처리한 내용): {name}", flush=True)
            continue
        print(f"문서 처리 중: {name}", flush=True)
        await rag.ainsert(text, file_paths=name)
        counts = await rag.doc_status.get_status_counts()
        if any(int(value) for key, value in counts.items() if key != "processed"):
            raise RuntimeError(f"문서 처리 실패/미완료: {counts}. 새 WORKING_DIR로 재시도하세요.")
        manifest[name] = hashes[name]
        temporary = manifest_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(manifest_file)
    counts = await rag.doc_status.get_status_counts()
    print("문서 처리 상태:", counts, flush=True)
    # ainsert can record failures instead of raising; never silently report success.
    if any(int(value) for key, value in counts.items() if key != "processed"):
        raise RuntimeError(f"색인이 완료되지 않았습니다: {counts}. 위 LightRAG 로그를 확인하세요.")
    return counts


async def require_index(rag: LightRAG):
    counts = await rag.doc_status.get_status_counts()
    if not int(counts.get("processed", 0)):
        raise RuntimeError("색인이 없습니다. 먼저 python app.py ingest 를 실행하세요.")


async def ask(rag: LightRAG, question: str, mode: str, context_only=False):
    if mode not in MODES:
        raise ValueError(f"지원하지 않는 모드: {mode}")
    await require_index(rag)
    started = time.perf_counter()
    result = await rag.aquery(question, param=QueryParam(
        mode=mode, only_need_context=context_only, stream=False,
        top_k=12, chunk_top_k=8,
        max_entity_tokens=2000, max_relation_tokens=2500, max_total_tokens=7000,
        enable_rerank=False, include_references=True, user_prompt=RULES,
        response_type="간결한 한국어 답변과 근거 및 관계 연결 경로",
    ))
    return {"mode": mode, "seconds": round(time.perf_counter() - started, 2),
            "result": result}


def graph_data():
    import networkx as nx
    paths = list(STORAGE.rglob("*.graphml"))
    if not paths:
        return {"nodes": [], "edges": [], "message": "아직 그래프 파일이 없습니다."}
    graph = nx.read_graphml(paths[0])
    return {"nodes": [{"id": str(n), **dict(d)} for n, d in graph.nodes(data=True)],
            "edges": [{"source": str(s), "target": str(t), **dict(d)}
                      for s, t, d in graph.edges(data=True)]}


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag = await create_rag()
    app.state.rag = rag
    app.state.lock = asyncio.Lock()
    try:
        yield
    finally:
        await rag.finalize_storages()


app = FastAPI(title="Gemma 3 + LightRAG 실습", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=3000)
    mode: str = "mix"
    context_only: bool = False


@app.get("/", response_class=HTMLResponse)
async def home():
    return (ROOT / "index.html").read_text(encoding="utf-8")


@app.get("/api/questions")
async def questions():
    return json.loads((ROOT / "questions.json").read_text(encoding="utf-8"))


@app.get("/healthz")
async def health():
    # Local process health; a remote Ollama outage must not cause restart loops.
    return {"status": "ok"}


@app.get("/readyz")
async def ready():
    counts = await app.state.rag.doc_status.get_status_counts()
    if not int(counts.get("processed", 0)):
        raise HTTPException(503, "문서 색인이 아직 없습니다.")
    return {"status": "ready", "documents": counts}


@app.get("/api/graph")
async def graph():
    return graph_data()


@app.post("/api/ask")
async def web_ask(body: AskRequest):
    if body.mode not in MODES:
        raise HTTPException(400, "지원하지 않는 검색 모드")
    if not body.question.strip():
        raise HTTPException(400, "질문을 입력하세요.")
    try:
        async with app.state.lock:
            return await ask(app.state.rag, body.question.strip(), body.mode, body.context_only)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


async def cli(args):
    if args.command == "graph":
        print(json.dumps(graph_data(), ensure_ascii=False, indent=2))
        return
    if args.command == "check":
        dim = await check_ollama()
        print(f"연결 성공: {HOST}\n답변/관계 추출: {MODEL}\n임베딩: {EMBED}, {dim}차원")
        return
    rag = await create_rag()
    try:
        if args.command == "ingest":
            await ingest(rag)
        elif args.command == "ask":
            print(json.dumps(await ask(rag, args.question, args.mode, args.context), ensure_ascii=False, indent=2))
        elif args.command == "compare":
            for mode in ("naive", "hybrid", "mix"):
                print(json.dumps(await ask(rag, args.question, mode, args.context), ensure_ascii=False, indent=2), flush=True)
    finally:
        await rag.finalize_storages()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma 3 12B + LightRAG 한국어 실습")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "ingest", "graph"):
        sub.add_parser(name)
    for name in ("ask", "compare"):
        p = sub.add_parser(name)
        p.add_argument("question")
        p.add_argument("--context", action="store_true", help="답변 대신 실제 검색 근거 출력")
        if name == "ask":
            p.add_argument("--mode", choices=sorted(MODES), default="mix")
    args = parser.parse_args()
    try:
        asyncio.run(cli(args))
    except (RuntimeError, ValueError, httpx.HTTPError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        sys.exit(1)
