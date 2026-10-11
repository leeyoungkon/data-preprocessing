import re
import pymupdf
import requests
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 설정
# --------------------------------------------------

PDF_FILE = "workplan.pdf"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3:12b"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_documents"


# --------------------------------------------------
# 1. PDF Parsing
# --------------------------------------------------

def extract_pdf(pdf_file):

    doc = pymupdf.open(pdf_file)

    pages = []

    for page_number, page in enumerate(doc):

        text = page.get_text("text", sort=True)

        pages.append({
            "page": page_number + 1,
            "text": text
        })

    doc.close()

    return pages


# --------------------------------------------------
# 2. Text Cleaning
# --------------------------------------------------

def clean_text(text):

    # 탭 제거
    text = text.replace("\t", " ")

    # 여러 개의 공백을 하나로
    text = re.sub(r"[ ]+", " ", text)

    # 지나치게 많은 줄바꿈 제거
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# 3. Chunking
# --------------------------------------------------

def create_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    chunk_id = 0

    for page in pages:

        cleaned_text = clean_text(page["text"])

        # 텍스트가 없는 페이지 제외
        if not cleaned_text:
            continue

        page_chunks = splitter.split_text(cleaned_text)

        for text in page_chunks:

            chunks.append({
                "id": f"chunk-{chunk_id}",
                "text": text,
                "page": page["page"],
                "source": PDF_FILE
            })

            chunk_id += 1

    return chunks


# --------------------------------------------------
# 4. Embedding 생성
# --------------------------------------------------

def create_embedding(text):

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": text
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["embeddings"][0]


# --------------------------------------------------
# 5. ChromaDB 연결
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# --------------------------------------------------
# 6. PDF → Chunk → Embedding → Vector DB
# --------------------------------------------------

def build_vector_db():

    print("PDF Parsing 시작...")

    pages = extract_pdf(PDF_FILE)

    print("PDF 페이지 수:", len(pages))

    print("Chunking 시작...")

    chunks = create_chunks(pages)

    print("생성된 Chunk 수:", len(chunks))

    print("Embedding 및 Vector DB 저장 시작...")

    for i, chunk in enumerate(chunks):

        embedding = create_embedding(
            chunk["text"]
        )

        collection.upsert(

            ids=[
                chunk["id"]
            ],

            documents=[
                chunk["text"]
            ],

            embeddings=[
                embedding
            ],

            metadatas=[
                {
                    "source": chunk["source"],
                    "page": chunk["page"]
                }
            ]
        )

        print(
            f"{i + 1}/{len(chunks)} 저장 완료 "
            f"(page={chunk['page']})"
        )

    print()
    print("Vector DB 저장 완료")


# --------------------------------------------------
# 7. 질문과 유사한 Chunk 검색
# --------------------------------------------------

def search_documents(question, n_results=3):

    # 질문 Embedding
    query_embedding = create_embedding(
        question
    )

    # Vector DB 검색
    results = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=n_results
    )

    return results


# --------------------------------------------------
# 8. 검색 결과 → Context 생성
# --------------------------------------------------

def create_context(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_parts = []

    for i in range(len(documents)):

        page = metadatas[i]["page"]
        text = documents[i]

        context_parts.append(
            f"[출처: {PDF_FILE}, 페이지: {page}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    return context


# --------------------------------------------------
# 9. Gemma3에게 Context + 질문 전달
# --------------------------------------------------

def ask_gemma(question, context):

    prompt = f"""
당신은 문서 기반 질의응답 시스템입니다.

아래 제공된 문서 내용만을 참고하여 질문에 답하세요.

문서에 답을 찾을 수 없는 경우에는
"제공된 문서에서 관련 내용을 찾을 수 없습니다."
라고 답하세요.

답변은 가능한 한 정확하고 간결하게 작성하세요.

----------------------------------------
[문서 Context]

{context}

----------------------------------------
[사용자 질문]

{question}

----------------------------------------
[답변]
"""

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )

    response.raise_for_status()

    data = response.json()

    return data["response"]


# --------------------------------------------------
# 10. RAG 질문 처리
# --------------------------------------------------

def rag(question):

    # STEP 1
    # 질문과 유사한 문서 검색
    results = search_documents(
        question,
        n_results=3
    )

    # STEP 2
    # 검색된 Chunk를 Context로 구성
    context = create_context(
        results
    )

    # STEP 3
    # Gemma3에게 Context + 질문 전달
    answer = ask_gemma(
        question,
        context
    )

    return answer, results


# --------------------------------------------------
# 11. 프로그램 시작
# --------------------------------------------------

print()
print("====================================")
print(" PDF RAG System")
print("====================================")


# Vector DB가 비어 있으면 PDF 처리
if collection.count() == 0:

    print()
    print("Vector DB가 비어 있습니다.")
    print("PDF를 처리합니다.")
    print()

    build_vector_db()

else:

    print()
    print(
        "기존 Vector DB를 사용합니다."
    )

    print(
        "저장된 Chunk 수:",
        collection.count()
    )


# --------------------------------------------------
# 12. 반복 질문
# --------------------------------------------------

while True:

    print()

    question = input(
        "질문을 입력하세요(exit 입력 시 종료): "
    )

    if question.lower() == "exit":
        break

    try:

        answer, results = rag(question)

        print()
        print("====================================")
        print("검색된 문서")
        print("====================================")

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for i in range(len(documents)):

            print()
            print(
                f"[검색 결과 {i + 1}]"
            )

            print(
                "페이지:",
                metadatas[i]["page"]
            )

            print(
                "거리:",
                round(distances[i], 4)
            )

            print(
                documents[i][:300]
            )

            print("...")

        print()
        print("====================================")
        print("Gemma3 답변")
        print("====================================")

        print(answer)

    except Exception as e:

        print()
        print(
            "오류가 발생했습니다:",
            e
        )