import requests

chunks = [
    "인공지능은 인간의 지능을 컴퓨터로 구현하는 기술이다.",
    "머신러닝은 데이터를 통해 패턴을 학습한다.",
    "PostgreSQL은 관계형 데이터베이스이다."
]

for chunk in chunks:

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": chunk
        }
    )

    embedding = response.json()["embeddings"][0]

    print(chunk)
    print(len(embedding))
    print()