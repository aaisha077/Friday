"""Ask the ATT&CK index: which techniques match this alert?"""
import sys
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="attack_db").get_collection("mitre_attack")

def lookup(alert_text, n=3):
    res = col.query(query_embeddings=model.encode([alert_text]).tolist(), n_results=n)
    return [dict(m, distance=round(d, 3)) for m, d in zip((res["metadatas"] or [[]])[0], (res["distances"] or [[]])[0])]

if __name__ == "__main__":
    alert = " ".join(sys.argv[1:]) or "Wazuh: Successful sudo to ROOT executed. User successfully changed UID."
    print(f"Alert: {alert}\n")
    for t in lookup(alert):
        print(f"{t['id']:10} {t['name']:45} [{t['tactics']}]  dist={t['distance']}")
        print(f"           {t['url']}")
