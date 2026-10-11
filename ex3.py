import requests

text = "인공지능은 인간의 지능을 컴퓨터로 구현하는 기술이다."

response = requests.post(
    "http://localhost:11434/api/embed",
    json={
        "model": "nomic-embed-text",
        "input": text
    }
)

data = response.json()

embedding = data["embeddings"][0]

print(embedding)
print("Vector dimension:", len(embedding))
