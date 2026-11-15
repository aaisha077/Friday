"""Score ATT&CK technique mapping on 39 real attack windows from AIT-ADS.
Ground truth = what the attacker was actually doing (AIT's per-event labels).
Compares: naive RAG (search raw alert, take top hit) vs the FRIDAY agent (rewrite + hybrid retrieval + pick)."""
import json, re, collections
from attack_lookup import lookup
from explain_alert import explain

cases = json.load(open("scripts/technique_test.json"))

def correct(tid, expected):
    return any(tid == e or tid.startswith(e + ".") for e in expected)

score = collections.defaultdict(lambda: [0, 0])     # method -> [right, total]
per_label = collections.defaultdict(lambda: collections.Counter())

for i, c in enumerate(cases, 1):
    naive = lookup(c["alert"], n=1)[0]["id"]
    _, _, answer = explain(c["alert"])
    m = re.search(r"T\d{4}(?:\.\d{3})?", answer)
    agent = m.group(0) if m else "none"

    for name, tid in [("naive RAG", naive), ("FRIDAY agent", agent)]:
        ok = correct(tid, c["expected"])
        score[name][0] += ok; score[name][1] += 1
        per_label[c["label"]][name] += ok
    per_label[c["label"]]["n"] += 1
    print(f"{i:2}/{len(cases)} {c['label']:22} expected={'/'.join(c['expected']):18} naive={naive:10} agent={agent}")

print("\nTechnique accuracy:")
for name, (right, total) in score.items():
    print(f"  {name:13} {right}/{total} = {right/total:.0%}")
print("\nPer attack step (naive / agent / cases):")
for lab, c in per_label.items():
    print(f"  {lab:22} {c['naive RAG']} / {c['FRIDAY agent']} / {c['n']}")
