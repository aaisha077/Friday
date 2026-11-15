"""Step 1 of the FRIDAY agent: put every MITRE ATT&CK technique into a Chroma vector DB,
so FRIDAY can look up "which attack technique does this alert look like?"."""
import json, os, re, urllib.request
import chromadb
from sentence_transformers import SentenceTransformer

URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
PATH = "data/enterprise-attack.json"

if not os.path.exists(PATH):
    print("Downloading MITRE ATT&CK (~50 MB)...")
    urllib.request.urlretrieve(URL, PATH)

objects = json.load(open(PATH))["objects"]
techniques = [o for o in objects
              if o["type"] == "attack-pattern" and not o.get("revoked") and not o.get("x_mitre_deprecated")]

ids, docs, metas = [], [], []
for t in techniques:
    tid = next(r["external_id"] for r in t["external_references"] if r.get("source_name") == "mitre-attack")
    tactics = ", ".join(p["phase_name"] for p in t.get("kill_chain_phases", []))
    desc = re.sub(r"\(Citation:[^)]*\)", "", t.get("description", ""))   # drop citation noise
    desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", desc)                   # markdown links -> text
    ids.append(tid)
    docs.append(f"{t['name']}. {desc[:1500]}")
    metas.append({"id": tid, "name": t["name"], "tactics": tactics,
                  "url": f"https://attack.mitre.org/techniques/{tid.replace('.', '/')}"})

print(f"Embedding {len(docs)} techniques...")
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(docs, show_progress_bar=True).tolist()

db = chromadb.PersistentClient(path="attack_db")
col = db.get_or_create_collection("mitre_attack")
col.upsert(ids=ids, documents=docs, embeddings=vectors, metadatas=metas)
print(f"Done: {col.count()} techniques in attack_db/")
