import re
import pymupdf
import requests
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==================================================
# 설정
# ==================================================

PDF_FILE = "workplan.pdf"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

# 한국어 검색 성능 개선
EMBEDDING_MODEL = "bge-m3"

LLM_MODEL = "gemma3:12b"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_documents_bge_m3"


# ==================================================
# 1. PDF Parsing
# ==================================================

def extract_pdf(pdf_file):

    doc = pymupdf.open(pdf_file)

    pages = []

    for page_number, page in enumerate(doc):

        text = page.get_text(
            "text",
            sort=True
        )

        pages.append({
            "page": page_number + 1,
            "text": text
        })

    doc.close()

    return pages


# ==================================================
# 2. Text Cleaning
# ==================================================

def clean_text(text):

    # 특수 줄바꿈 문자 정리
    text = text.replace("\r", "\n")

    # 탭 → 공백
    text = text.replace("\t", " ")

    # 여러 공백 제거
    text = re.sub(
        r"[ ]{2,}",
        " ",
        text
    )

    # PDF에서 단어 중간에 줄이 바뀐 경우 완화
    text = re.sub(
        r"(?<=[가-힣A-Za-z0-9])\n(?=[가-힣A-Za-z0-9])",
        " ",
        text
    )

    # 과도한 줄바꿈 정리
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ==================================================
# 3. Chunking
# ==================================================

def create_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(

        # 기존 1000 → 1800
        chunk_size=1800,

        # 문맥 보존
        chunk_overlap=300,

        separators=[
            "\n\n",
            "\n",
            "。 ",
            ". ",
            "다. ",
            "요. ",
            " ",
            ""
        ]
    )

    chunks = []

    chunk_id = 0

    for page in pages:

        cleaned_text = clean_text(
            page["text"]
        )

        if len(cleaned_text) < 20:
            continue

        page_chunks = splitter.split_text(
            cleaned_text
        )

        for index, text in enumerate(page_chunks):

            chunks.append({

                "id":
                    f"page-{page['page']}-chunk-{index}",

                "text":
                    text,

                "page":
                    page["page"],

                "source":
                    PDF_FILE
            })

            chunk_id += 1

    return chunks


# ==================================================
# 4. Embedding
# ==================================================

def create_embedding(text):

    response = requests.post(

        OLLAMA_EMBED_URL,

        json={
            "model":
                EMBEDDING_MODEL,

            "input":
                text
        },

        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["embeddings"][0]


# ==================================================
# 5. ChromaDB
# ==================================================

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ==================================================
# 6. Vector DB 생성
# ==================================================

def build_vector_db():

    print()
    print("PDF Parsing 시작...")

    pages = extract_pdf(
        PDF_FILE
    )

    print(
        "PDF 페이지 수:",
        len(pages)
    )

    print()
    print("Chunking 시작...")

    chunks = create_chunks(
        pages
    )

    print(
        "생성된 Chunk 수:",
        len(chunks)
    )

    print()
    print(
        "Embedding 및 Vector DB 저장 시작..."
    )

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
                    "source":
                        chunk["source"],

                    "page":
                        chunk["page"]
                }
            ]
        )

        print(
            f"{i + 1}/{len(chunks)}",
            f"page={chunk['page']}"
        )

    print()
    print(
        "Vector DB 저장 완료"
    )


# ==================================================
# 7. Vector 검색
# ==================================================

def search_documents(question):

    query_embedding = create_embedding(
        question
    )

    # 처음부터 3개만 뽑지 않고
    # 8개 정도 후보를 가져온다.
    results = collection.query(

        query_embeddings=[
            query_embedding
        ],

        n_results=8
    )

    return results


# ==================================================
# 8. Context 생성
# ==================================================

def create_context(results):

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    context_parts = []

    # 상위 5개 문서를 실제 LLM Context로 사용
    for i in range(
        min(5, len(documents))
    ):

        page = metadatas[i]["page"]
        text = documents[i]

        context_parts.append(
            f"""
[문서 {i + 1}]
출처 페이지: {page}

{text}
"""
        )

    return "\n".join(
        context_parts
    )


# ==================================================
# 9. Gemma3
# ==================================================

def ask_gemma(
    question,
    context
):

    prompt = f"""
당신은 제안요청서와 사업 문서를 분석하는 전문 어시스턴트입니다.

아래의 [참고 문서]를 바탕으로 사용자의 질문에
자연스러운 한국어로 답변하세요.

규칙:

1. 여러 문서 조각에 흩어진 내용을 종합해서 설명하세요.

2. 질문에 등장하는 용어가 문서에 직접 정의되어 있지 않더라도,
   문서의 주변 내용과 사용 맥락을 통해 의미를 파악할 수 있다면
   그 의미를 설명하세요.

3. 먼저 질문에 대한 핵심 답을 말하고,
   필요한 경우 세부 내용을 설명하세요.

4. 문서에서 확인할 수 있는 내용과
   일반적인 설명을 구분하세요.

5. 근거가 있는 경우 관련 페이지를 함께 표시하세요.

6. 단순히 "관련 내용을 찾을 수 없습니다"라고 답하지 말고,
   제공된 문서에서 확인 가능한 범위까지 최대한 설명하세요.

7. 답변은 자연스러운 한국어 문장으로 작성하세요.


==============================
[참고 문서]
==============================

{context}


==============================
[사용자 질문]
==============================

{question}


==============================
[답변]
==============================
"""

    response = requests.post(

        OLLAMA_GENERATE_URL,

        json={
            "model":
                LLM_MODEL,

            "prompt":
                prompt,

            "stream":
                False,

            "options": {
                "temperature": 0.2
            }
        },

        timeout=300
    )

    response.raise_for_status()

    return response.json()[
        "response"
    ]


# ==================================================
# 10. RAG
# ==================================================

def rag(question):

    results = search_documents(
        question
    )

    context = create_context(
        results
    )

    answer = ask_gemma(
        question,
        context
    )

    return (
        answer,
        results,
        context
    )


# ==================================================
# 11. 시작
# ==================================================

print()
print(
    "===================================="
)
print(
    " PDF RAG System"
)
print(
    "===================================="
)


if collection.count() == 0:

    print()
    print(
        "Vector DB가 비어 있습니다."
    )

    build_vector_db()

else:

    print()
    print(
        "기존 Vector DB 사용"
    )

    print(
        "Chunk 수:",
        collection.count()
    )


# ==================================================
# 12. 반복 질문
# ==================================================

while True:

    print()

    question = input(
        "질문을 입력하세요(exit 입력 시 종료): "
    )

    if question.lower() == "exit":
        break

    try:

        (
            answer,
            results,
            context
        ) = rag(question)

        print()
        print(
            "===================================="
        )

        print(
            "검색된 문서"
        )

        print(
            "===================================="
        )

        documents = \
            results["documents"][0]

        metadatas = \
            results["metadatas"][0]

        distances = \
            results["distances"][0]

        # 확인을 위해 상위 5개 출력
        for i in range(
            min(5, len(documents))
        ):

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
                round(
                    distances[i],
                    4
                )
            )

            print()

            print(
                documents[i][:700]
            )

            print(
                "..."
            )

        print()
        print(
            "===================================="
        )

        print(
            "Gemma3 답변"
        )

        print(
            "===================================="
        )

        print()

        print(
            answer
        )

    except Exception as e:

        print()
        print(
            "오류:",
            e
        )