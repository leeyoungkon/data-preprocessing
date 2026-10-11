from pathlib import Path
import hashlib

import chromadb
import requests

# 텍스트 파일은 이 Python 파일과 같은 폴더에 둡니다.
BASE_DIR = Path(__file__).resolve().parent
TEXT_FILE = BASE_DIR / "ai_overview.txt"
DB_PATH = BASE_DIR / "chroma_db"
OLLAMA_URL = "http://localhost:11434"
MODEL = "nomic-embed-text"
CHUNK_SIZE = 500   # 토큰 수가 아닌 문자 수
CHUNK_OVERLAP = 100


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not 0 <= overlap < chunk_size:
        raise ValueError("0 <= overlap < chunk_size 조건이 필요합니다.")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
    return chunks


def main():
    # utf-8-sig는 UTF-8과 BOM이 있는 UTF-8 파일을 모두 읽습니다.
    text = TEXT_FILE.read_text(encoding="utf-8-sig")
    chunks = split_text(text)
    if not chunks:
        raise ValueError("텍스트 파일에 내용이 없습니다.")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_or_create_collection(
        name="documents", embedding_function=None
    )
    # 파일별로 다른 ID를 사용합니다. 같은 경로의 파일 재실행은 갱신입니다.
    source_key = hashlib.sha256(str(TEXT_FILE.resolve()).encode("utf-8")).hexdigest()

    with requests.Session() as session:
        for i, chunk in enumerate(chunks):
            response = session.post(
                f"{OLLAMA_URL.rstrip('/')}/api/embed",
                json={"model": MODEL, "input": chunk, "truncate": False},
                timeout=180,
            )
            if not response.ok:
                raise RuntimeError(
                    f"Ollama 오류 ({response.status_code}): {response.text}"
                )
            vectors = response.json().get("embeddings")
            if not isinstance(vectors, list) or len(vectors) != 1 or not vectors[0]:
                raise RuntimeError("Ollama 응답에 유효한 임베딩이 없습니다.")

            collection.upsert(
                ids=[f"{source_key}-chunk-{i}"],
                documents=[chunk],
                embeddings=vectors,
                metadatas=[{
                    "source": str(TEXT_FILE.resolve()),
                    "source_key": source_key,
                    "chunk_index": i,
                    "embedding_model": MODEL,
                }],
            )
            print(f"저장 완료: {i + 1}/{len(chunks)} 청크 ({len(chunk)}자)")

    # 파일이 짧아졌을 때 이전 실행의 남은 청크를 제거합니다.
    # 모든 새 청크의 저장이 성공한 후에만 수행합니다.
    collection.delete(where={"$and": [
        {"source_key": source_key},
        {"chunk_index": {"$gte": len(chunks)}},
    ]})
    print(f"파일: {TEXT_FILE}\n청크 수: {len(chunks)}\nDB 위치: {DB_PATH}")


if __name__ == "__main__":
    main()
