import re
import math
import pymupdf
import requests
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 설정
# ============================================================

PDF_FILE = "workplan.pdf"

OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

EMBEDDING_MODEL = "bge-m3"
LLM_MODEL = "gemma3:12b"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "pdf_hybrid_table"

# Hybrid Search 가중치
VECTOR_WEIGHT = 0.6
KEYWORD_WEIGHT = 0.4

# 최종 검색 개수
FINAL_RESULTS = 3


# ============================================================
# 1. 일반 텍스트 Cleaning
# ============================================================

def clean_text(text):

    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")

    # 여러 공백 → 하나
    text = re.sub(
        r"[ ]{2,}",
        " ",
        text
    )

    # PDF에서 단어 사이에 불필요하게 줄이 끊긴 경우 완화
    text = re.sub(
        r"(?<=[가-힣A-Za-z0-9])\n(?=[가-힣A-Za-z0-9])",
        " ",
        text
    )

    # 너무 많은 줄바꿈 제거
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# 2. Table 셀 Cleaning
# ============================================================

def clean_table_cell(value):

    if value is None:
        return ""

    value = str(value)

    value = value.replace("\n", " ")
    value = value.replace("\t", " ")

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# 3. Table → 구조화된 Markdown 변환
# ============================================================

def table_to_markdown(table):

    try:

        rows = table.extract()

    except Exception:

        return ""

    if not rows:
        return ""

    cleaned_rows = []

    for row in rows:

        if row is None:
            continue

        cleaned_row = [
            clean_table_cell(cell)
            for cell in row
        ]

        # 완전히 빈 행 제거
        if not any(cleaned_row):
            continue

        cleaned_rows.append(
            cleaned_row
        )

    if not cleaned_rows:
        return ""

    # --------------------------------------------------------
    # 열 개수 맞추기
    # --------------------------------------------------------

    max_columns = max(
        len(row)
        for row in cleaned_rows
    )

    normalized_rows = []

    for row in cleaned_rows:

        row = row + (
            [""] *
            (max_columns - len(row))
        )

        normalized_rows.append(
            row
        )

    # --------------------------------------------------------
    # 첫 번째 행을 Header로 사용
    # --------------------------------------------------------

    header = normalized_rows[0]

    # 헤더가 완전히 비어있는 경우
    if not any(header):

        header = [
            f"컬럼{i + 1}"
            for i in range(max_columns)
        ]

        data_rows = normalized_rows

    else:

        # 비어있는 헤더명 보완
        header = [
            cell if cell
            else f"컬럼{i + 1}"

            for i, cell in enumerate(
                header
            )
        ]

        data_rows = normalized_rows[1:]

    # --------------------------------------------------------
    # Markdown
    # --------------------------------------------------------

    lines = []

    lines.append(
        "| " +
        " | ".join(header) +
        " |"
    )

    lines.append(
        "| " +
        " | ".join(
            ["---"] * max_columns
        ) +
        " |"
    )

    for row in data_rows:

        lines.append(
            "| " +
            " | ".join(row) +
            " |"
        )

    return "\n".join(lines)


# ============================================================
# 4. 두 Rectangle이 겹치는지 확인
# ============================================================

def rect_intersects(
    rect1,
    rect2
):

    r1 = pymupdf.Rect(rect1)
    r2 = pymupdf.Rect(rect2)

    intersection = r1 & r2

    if intersection.is_empty:
        return False

    # 블록 면적
    area1 = (
        r1.width *
        r1.height
    )

    if area1 <= 0:
        return False

    # 블록에서 표 영역이 차지하는 비율
    intersection_area = (
        intersection.width *
        intersection.height
    )

    ratio = (
        intersection_area /
        area1
    )

    # 블록의 상당 부분이 표와 겹치면 표 텍스트로 판단
    return ratio > 0.3


# ============================================================
# 5. 페이지에서 Table 탐지
# ============================================================

def detect_tables(page):

    try:

        tables = page.find_tables()

        if tables and tables.tables:

            return tables.tables

    except Exception as e:

        print(
            "기본 Table 탐지 오류:",
            e
        )

    # --------------------------------------------------------
    # 선 없는 표일 경우 Text 기반 탐지 재시도
    # --------------------------------------------------------

    try:

        tables = page.find_tables(
            strategy="text"
        )

        if tables and tables.tables:

            return tables.tables

    except Exception:

        pass

    return []


# ============================================================
# 6. PDF Parsing
#
# 일반 본문과 Table을 분리한다.
# ============================================================

def extract_pdf(pdf_file):

    doc = pymupdf.open(
        pdf_file
    )

    elements = []

    print()
    print(
        "PDF 페이지 수:",
        len(doc)
    )

    for page_number, page in enumerate(
        doc
    ):

        page_no = page_number + 1

        print(
            f"Page {page_no} 처리 중..."
        )

        # ----------------------------------------------------
        # 6-1. Table 탐지
        # ----------------------------------------------------

        tables = detect_tables(
            page
        )

        table_rects = []

        # ----------------------------------------------------
        # 6-2. Table 별도 추출
        # ----------------------------------------------------

        for table_index, table in enumerate(
            tables
        ):

            table_rects.append(
                table.bbox
            )

            table_text = (
                table_to_markdown(
                    table
                )
            )

            if len(table_text) < 10:
                continue

            elements.append({

                "page":
                    page_no,

                "type":
                    "table",

                "table_index":
                    table_index + 1,

                "text":
                    table_text
            })

        # ----------------------------------------------------
        # 6-3. 페이지 Text Block 추출
        # ----------------------------------------------------

        blocks = page.get_text(
            "blocks",
            sort=True
        )

        for block in blocks:

            # block:
            # x0, y0, x1, y1, text, block_no, block_type

            if len(block) < 5:
                continue

            bbox = (
                block[0],
                block[1],
                block[2],
                block[3]
            )

            text = block[4]

            # ------------------------------------------------
            # 해당 block이 Table 영역인지 확인
            # ------------------------------------------------

            inside_table = False

            for table_rect in table_rects:

                if rect_intersects(
                    bbox,
                    table_rect
                ):

                    inside_table = True
                    break

            # 표 내용은 본문에서 제거
            if inside_table:
                continue

            text = clean_text(
                text
            )

            if len(text) < 20:
                continue

            elements.append({

                "page":
                    page_no,

                "type":
                    "text",

                "table_index":
                    0,

                "text":
                    text
            })

    doc.close()

    return elements


# ============================================================
# 7. Chunking
# ============================================================

def create_chunks(elements):

    # --------------------------------------------------------
    # 본문용 Chunker
    # --------------------------------------------------------

    text_splitter = (
        RecursiveCharacterTextSplitter(

            chunk_size=1800,

            chunk_overlap=300,

            separators=[
                "\n\n",
                "\n",
                "다. ",
                ". ",
                " ",
                ""
            ]
        )
    )

    # --------------------------------------------------------
    # 표는 조금 크게 유지
    #
    # 작은 표라면 가능한 한 한 Chunk에 보존
    # --------------------------------------------------------

    table_splitter = (
        RecursiveCharacterTextSplitter(

            chunk_size=2500,

            chunk_overlap=300,

            separators=[
                "\n",
                " ",
                ""
            ]
        )
    )

    chunks = []

    chunk_count = 0

    for element in elements:

        text = element["text"]

        # ----------------------------------------------------
        # Content Type에 따라 다른 Chunking
        # ----------------------------------------------------

        if element["type"] == "table":

            split_texts = (
                table_splitter.split_text(
                    text
                )
            )

        else:

            split_texts = (
                text_splitter.split_text(
                    text
                )
            )

        for local_index, chunk_text in enumerate(
            split_texts
        ):

            chunk_id = (
                f"page-{element['page']}"
                f"-{element['type']}"
                f"-{chunk_count}"
            )

            chunks.append({

                "id":
                    chunk_id,

                "page":
                    element["page"],

                "type":
                    element["type"],

                "table_index":
                    element[
                        "table_index"
                    ],

                "source":
                    PDF_FILE,

                "text":
                    chunk_text
            })

            chunk_count += 1

    return chunks


# ============================================================
# 8. Embedding 생성
# ============================================================

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

    return data[
        "embeddings"
    ][0]


# ============================================================
# 9. ChromaDB
# ============================================================

client = (
    chromadb.PersistentClient(
        path=CHROMA_PATH
    )
)

collection = (
    client.get_or_create_collection(
        name=COLLECTION_NAME
    )
)


# ============================================================
# 10. Vector DB 구축
# ============================================================

def build_vector_db():

    print()
    print(
        "===================================="
    )

    print(
        "PDF Parsing 및 Table 추출"
    )

    print(
        "===================================="
    )

    elements = extract_pdf(
        PDF_FILE
    )

    text_elements = [
        e for e in elements
        if e["type"] == "text"
    ]

    table_elements = [
        e for e in elements
        if e["type"] == "table"
    ]

    print()
    print(
        "본문 요소:",
        len(text_elements)
    )

    print(
        "Table 요소:",
        len(table_elements)
    )

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    print()
    print(
        "Chunking 시작..."
    )

    chunks = create_chunks(
        elements
    )

    print(
        "생성된 Chunk:",
        len(chunks)
    )

    print()
    print(
        "Embedding + Vector DB 저장..."
    )

    # --------------------------------------------------------
    # 저장
    # --------------------------------------------------------

    for i, chunk in enumerate(
        chunks
    ):

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
                    "page":
                        chunk["page"],

                    "source":
                        chunk["source"],

                    "type":
                        chunk["type"],

                    "table_index":
                        chunk[
                            "table_index"
                        ]
                }
            ]
        )

        print(
            f"{i + 1}/{len(chunks)} "
            f"page={chunk['page']} "
            f"type={chunk['type']}"
        )

    print()
    print(
        "Vector DB 구축 완료"
    )


# ============================================================
# 11. 한국어 Keyword 전처리
# ============================================================

def tokenize(text):

    text = text.lower()

    words = re.findall(
        r"[가-힣A-Za-z0-9]+",
        text
    )

    stopwords = {

        "뭐야",
        "뭔가",
        "무엇",
        "무엇인가",

        "말해줘",
        "알려줘",
        "설명해줘",

        "대해",
        "대한",

        "설명",
        "알려",

        "어떤",
        "것",

        "있어",
        "있나요",

        "해줘"
    }

    # --------------------------------------------------------
    # 간단한 한국어 조사 제거
    # --------------------------------------------------------

    particles = [

        "으로부터",
        "에게서",

        "에서는",
        "으로는",

        "에서",
        "에게",
        "으로",

        "에는",
        "부터",
        "까지",

        "보다",
        "처럼",

        "하고",
        "이며",
        "이고",

        "의",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "에",
        "와",
        "과",
        "도",
        "로"
    ]

    result = []

    for word in words:

        if word in stopwords:
            continue

        for particle in particles:

            if (
                word.endswith(
                    particle
                )
                and
                len(word)
                > len(particle) + 1
            ):

                word = (
                    word[
                        :-len(
                            particle
                        )
                    ]
                )

                break

        if (
            len(word) > 1
            and
            word not in stopwords
        ):

            result.append(
                word
            )

    return result


# ============================================================
# 12. Keyword Score
# ============================================================

def keyword_score(
    question,
    document
):

    keywords = tokenize(
        question
    )

    if not keywords:
        return 0.0

    document_lower = (
        document.lower()
    )

    document_no_space = re.sub(
        r"\s+",
        "",
        document_lower
    )

    score = 0.0

    # --------------------------------------------------------
    # 개별 Keyword
    # --------------------------------------------------------

    matched_keywords = 0

    for keyword in keywords:

        count = document_lower.count(
            keyword
        )

        if count > 0:

            matched_keywords += 1

            # 기본 점수
            score += 1.0

            # 여러 번 등장
            score += math.log(
                count + 1
            )

    # --------------------------------------------------------
    # 질문 키워드가 여러 개이고
    # 다수가 동시에 포함된 경우 보너스
    # --------------------------------------------------------

    if len(keywords) > 1:

        coverage = (
            matched_keywords /
            len(keywords)
        )

        score += (
            coverage * 2.0
        )

    # --------------------------------------------------------
    # 구문 Exact Match
    # --------------------------------------------------------

    phrase = " ".join(
        keywords
    )

    no_space_phrase = "".join(
        keywords
    )

    if phrase in document_lower:

        score += 4.0

    if (
        no_space_phrase
        in document_no_space
    ):

        score += 6.0

    return score


# ============================================================
# 13. Vector Search
# ============================================================

def vector_search(
    question,
    n_results=20
):

    embedding = create_embedding(
        question
    )

    results = collection.query(

        query_embeddings=[
            embedding
        ],

        n_results=n_results
    )

    vector_results = []

    for i in range(
        len(
            results["ids"][0]
        )
    ):

        vector_results.append({

            "id":
                results["ids"][0][i],

            "text":
                results[
                    "documents"
                ][0][i],

            "page":
                results[
                    "metadatas"
                ][0][i]["page"],

            "type":
                results[
                    "metadatas"
                ][0][i]["type"],

            "table_index":
                results[
                    "metadatas"
                ][0][i].get(
                    "table_index",
                    0
                ),

            "distance":
                results[
                    "distances"
                ][0][i]
        })

    return vector_results


# ============================================================
# 14. 모든 Chunk 조회
#     Keyword 검색용
# ============================================================

def get_all_chunks():

    data = collection.get(

        include=[
            "documents",
            "metadatas"
        ]
    )

    chunks = []

    for i in range(
        len(data["ids"])
    ):

        metadata = (
            data[
                "metadatas"
            ][i]
        )

        chunks.append({

            "id":
                data["ids"][i],

            "text":
                data[
                    "documents"
                ][i],

            "page":
                metadata[
                    "page"
                ],

            "type":
                metadata[
                    "type"
                ],

            "table_index":
                metadata.get(
                    "table_index",
                    0
                )
        })

    return chunks


# ============================================================
# 15. Keyword Search
# ============================================================

def keyword_search(
    question,
    n_results=20
):

    chunks = get_all_chunks()

    results = []

    for chunk in chunks:

        score = keyword_score(
            question,
            chunk["text"]
        )

        if score <= 0:
            continue

        results.append({

            **chunk,

            "keyword_raw":
                score
        })

    results.sort(

        key=lambda x:
            x[
                "keyword_raw"
            ],

        reverse=True
    )

    return results[
        :n_results
    ]


# ============================================================
# 16. Vector Score 정규화
# ============================================================

def normalize_vector(
    results
):

    if not results:
        return results

    distances = [
        item["distance"]
        for item in results
    ]

    min_d = min(
        distances
    )

    max_d = max(
        distances
    )

    diff = (
        max_d -
        min_d
    )

    if diff == 0:
        diff = 1

    for result in results:

        result[
            "vector_score"
        ] = (
            1 -
            (
                result["distance"]
                - min_d
            )
            / diff
        )

    return results


# ============================================================
# 17. Keyword Score 정규화
# ============================================================

def normalize_keyword(
    results
):

    if not results:
        return results

    max_score = max(

        item[
            "keyword_raw"
        ]

        for item in results
    )

    if max_score == 0:
        max_score = 1

    for result in results:

        result[
            "keyword_score"
        ] = (
            result[
                "keyword_raw"
            ]
            /
            max_score
        )

    return results


# ============================================================
# 18. Hybrid Search
# ============================================================

def hybrid_search(
    question,
    final_results=6
):

    # --------------------------------------------------------
    # Vector
    # --------------------------------------------------------

    vector_results = (
        normalize_vector(

            vector_search(
                question,
                n_results=20
            )
        )
    )

    # --------------------------------------------------------
    # Keyword
    # --------------------------------------------------------

    keyword_results = (
        normalize_keyword(

            keyword_search(
                question,
                n_results=20
            )
        )
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merged = {}

    for result in vector_results:

        merged[
            result["id"]
        ] = {

            "id":
                result["id"],

            "text":
                result["text"],

            "page":
                result["page"],

            "type":
                result["type"],

            "table_index":
                result[
                    "table_index"
                ],

            "vector_score":
                result[
                    "vector_score"
                ],

            "keyword_score":
                0.0
        }

    # --------------------------------------------------------
    # Keyword 결과 추가
    # --------------------------------------------------------

    for result in keyword_results:

        chunk_id = (
            result["id"]
        )

        if chunk_id not in merged:

            merged[
                chunk_id
            ] = {

                "id":
                    chunk_id,

                "text":
                    result["text"],

                "page":
                    result["page"],

                "type":
                    result["type"],

                "table_index":
                    result[
                        "table_index"
                    ],

                "vector_score":
                    0.0,

                "keyword_score":
                    result[
                        "keyword_score"
                    ]
            }

        else:

            merged[
                chunk_id
            ][
                "keyword_score"
            ] = result[
                "keyword_score"
            ]

    # --------------------------------------------------------
    # Hybrid Score
    # --------------------------------------------------------

    results = []

    for item in merged.values():

        hybrid_score = (

            VECTOR_WEIGHT
            *
            item[
                "vector_score"
            ]

            +

            KEYWORD_WEIGHT
            *
            item[
                "keyword_score"
            ]
        )

        item[
            "hybrid_score"
        ] = hybrid_score

        results.append(
            item
        )

    # --------------------------------------------------------
    # 높은 점수 순
    # --------------------------------------------------------

    results.sort(

        key=lambda x:
            x[
                "hybrid_score"
            ],

        reverse=True
    )

    return results[
        :final_results
    ]


# ============================================================
# 19. Context 생성
# ============================================================

def create_context(
    results
):

    contexts = []

    for i, result in enumerate(
        results
    ):

        # ----------------------------------------------------
        # 표인지 본문인지 명확히 알려줌
        # ----------------------------------------------------

        if result["type"] == "table":

            content_type = (
                f"표(Table)"
                f" #{result['table_index']}"
            )

        else:

            content_type = (
                "본문(Text)"
            )

        context = f"""
[자료 {i + 1}]
페이지: {result['page']}
자료 유형: {content_type}

{result['text']}
"""

        contexts.append(
            context
        )

    return "\n\n".join(
        contexts
    )


# ============================================================
# 20. Gemma3
# ============================================================

def ask_gemma(
    question,
    context
):

    prompt = f"""
당신은 공공기관 정보시스템 구축 제안요청서(RFP)를
분석하는 전문 어시스턴트입니다.

아래에 제공되는 자료에는 일반 본문과 표(Table)가
포함될 수 있습니다.

사용자의 질문에 대해 제공된 자료를 종합하여
자연스러운 한국어로 설명하세요.


[중요 규칙]

1. 질문에 대한 핵심 답변부터 제시하세요.

2. 본문과 표의 내용을 함께 참고하세요.

3. 표가 제공된 경우 단순히 행과 열을 그대로 읽어주지 말고
   표가 의미하는 내용을 자연어로 해석해서 설명하세요.

4. 질문의 내용이 여러 Chunk에 흩어져 있다면
   이를 종합하여 답변하세요.

5. 질문에 등장한 용어가 직접 정의되어 있지 않더라도
   문서의 사용 맥락으로 의미를 파악할 수 있다면
   그 의미를 설명하세요.

6. 문서에 명시된 사실과 일반적인 해석을
   가능한 한 구분하세요.

7. 근거가 있는 경우 해당 페이지를 표시하세요.

8. 검색 결과 중 질문과 관계없는 내용은 사용하지 마세요.

9. 문서에서 확인되지 않는 내용을
   사실인 것처럼 만들어내지 마세요.

10. 단순히 문장을 복사하지 말고
    사람이 설명하는 것처럼 자연스러운 한국어로 답하세요.


==================================================
[참고 문서]
==================================================

{context}


==================================================
[사용자 질문]
==================================================

{question}


==================================================
[답변]
==================================================
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

                "temperature":
                    0.2
            }
        },

        timeout=300
    )

    response.raise_for_status()

    return response.json()[
        "response"
    ]


# ============================================================
# 21. RAG
# ============================================================

def rag(question):

    # --------------------------------------------------------
    # Hybrid Search
    # --------------------------------------------------------

    results = hybrid_search(

        question,

        final_results=
            FINAL_RESULTS
    )

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context = create_context(
        results
    )

    # --------------------------------------------------------
    # Gemma3
    # --------------------------------------------------------

    answer = ask_gemma(

        question,
        context
    )

    return (
        answer,
        results
    )


# ============================================================
# 22. Main
# ============================================================

print()
print(
    "========================================"
)

print(
    " PDF Hybrid RAG + Table Processing"
)

print(
    "========================================"
)


# ------------------------------------------------------------
# DB 없으면 구축
# ------------------------------------------------------------

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
        "저장된 Chunk:",
        collection.count()
    )


# ============================================================
# 23. 반복 질문
# ============================================================

while True:

    print()

    question = input(
        "질문을 입력하세요(exit 입력 시 종료): "
    )

    if (
        question.lower()
        == "exit"
    ):

        break

    try:

        answer, results = rag(
            question
        )

        # ----------------------------------------------------
        # 검색 결과
        # ----------------------------------------------------

        print()
        print(
            "========================================"
        )

        print(
            "Hybrid 검색 결과"
        )

        print(
            "========================================"
        )

        for i, result in enumerate(
            results
        ):

            print()
            print(
                f"[검색 결과 {i + 1}]"
            )

            print(
                "페이지:",
                result["page"]
            )

            print(
                "자료 유형:",
                result["type"]
            )

            if (
                result["type"]
                == "table"
            ):

                print(
                    "Table:",
                    result[
                        "table_index"
                    ]
                )

            print(
                "Vector:",
                round(
                    result[
                        "vector_score"
                    ],
                    4
                )
            )

            print(
                "Keyword:",
                round(
                    result[
                        "keyword_score"
                    ],
                    4
                )
            )

            print(
                "Hybrid:",
                round(
                    result[
                        "hybrid_score"
                    ],
                    4
                )
            )

            print()

            print(
                result["text"][:1000]
            )

            print(
                "..."
            )

        # ----------------------------------------------------
        # 최종 답변
        # ----------------------------------------------------

        print()
        print(
            "========================================"
        )

        print(
            "Gemma3 답변"
        )

        print(
            "========================================"
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