import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="attack_memory_db")
memory = client.get_or_create_collection("attacks")

def save_attack(attack_id, goal, attack_text, worked):
    vector = model.encode([goal]).tolist()
    memory.upsert(
        ids=[attack_id],
        documents=[goal],
        embeddings=vector,
        metadatas=[{"attack_text": attack_text, "worked": worked}],
    )

def find_similar(goal, n=3):
    if memory.count() == 0:
        return []
    vector = model.encode([goal]).tolist()
    result = memory.query(query_embeddings=vector, n_results=n, where={"worked": True})
    found = []
    for meta, dist in zip(result["metadatas"][0], result["distances"][0]):
        found.append({"attack_text": meta["attack_text"], "distance": round(dist, 3)})
    return found