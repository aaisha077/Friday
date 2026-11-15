import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
memory = client.create_collection("attack_memory")

texts = [
    "Asked the assistant to summarize a file, and the file contained extra instructions",
    "Pretended to be the system administrator asking for a password reset",
    "Asked for a translation of a document into French",
]
ids = ["a1", "a2", "a3"]
vectors = model.encode(texts).tolist()
memory.add(ids=ids, documents=texts, embeddings=vectors)

question = "the agent read a bill that had hidden instructions in it"
q_vector = model.encode([question]).tolist()
result = memory.query(query_embeddings=q_vector, n_results=2)

print("Query:", question)
for doc, dist in zip(result["documents"][0], result["distances"][0]):
    print(round(dist, 3), "|", doc)