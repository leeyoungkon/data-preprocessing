import requests
import chromadb
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="documents"
)
question = "머신러닝은 어떻게 학습하나요?"

response = requests.post(
    "http://localhost:11434/api/embed",
    json={
        "model": "nomic-embed-text",
        "input": question
    }
)

query_embedding = response.json()["embeddings"][0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print(results["documents"])
