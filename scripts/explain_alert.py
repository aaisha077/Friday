"""FRIDAY agent v0: alert -> rewrite into attacker behaviour -> ATT&CK lookup -> explanation.
Raw alert text retrieves badly ("400 error x7000" matched WMI), so we first ask an LLM to
describe WHAT THE ATTACKER IS DOING, then search ATT&CK with that description."""
import json, os, re, sys
from openai import OpenAI
from dotenv import load_dotenv
from attack_lookup import lookup, col

load_dotenv()
llm = OpenAI(api_key=os.environ["GROQ_API_KEY"],
             base_url="https://api.groq.com/openai/v1", timeout=60, max_retries=2)
MODEL = "openai/gpt-oss-120b"

def ask(prompt):
    r = llm.chat.completions.create(model=MODEL, temperature=0,
                                    messages=[{"role": "user", "content": prompt}])
    return (r.choices[0].message.content or "").strip()

def explain(alert):
    # 1. rewrite: alert -> attacker behaviour in ATT&CK-style words
    behaviour = ask("In one sentence, describe what an attacker could be doing if a SOC saw this alert. "
                    "Use MITRE ATT&CK vocabulary (e.g. brute force, scanning, privilege escalation). "
                    "Counts like x400 mean that many events in 5 minutes. "
                    "End with the 2-3 most likely ATT&CK technique IDs, e.g. (T1110, T1078).\n"
                    f"Alert: {alert}")
    # 2. retrieve: (a) techniques the LLM named by ID, checked against the real ATT&CK DB
    #              (b) semantic search with the rewritten text
    named = set(re.findall(r"T\d{4}(?:\.\d{3})?", behaviour))
    named |= {t.split(".")[0] for t in named}            # also add the parent technique
    found = col.get(ids=sorted(named)) if named else {"metadatas": []}
    candidates = [dict(m, distance="named") for m in (found["metadatas"] or [])]
    seen = {str(c["id"]) for c in candidates}
    candidates += [c for c in lookup(behaviour, n=5) if c["id"] not in seen]
    options = "\n".join(f"{c['id']} {c['name']} [{c['tactics']}]" for c in candidates)
    # 3. pick + explain, only from the retrieved techniques
    answer = ask("You are a SOC analyst. Pick the ONE best matching technique from the list "
                 "(or 'none' if nothing fits) and explain in 2 sentences what is happening and what to do. "
                 "Pay attention to counts: many events from one source in 5 minutes usually means "
                 "automated scanning or brute force, not a single user action.\n"
                 f"Alert: {alert}\nCandidate techniques:\n{options}\n"
                 'Reply as JSON: {"technique_id": "...", "severity": "low|medium|high|critical", "explanation": "..."}')
    return behaviour, candidates, answer

if __name__ == "__main__":
    alert = " ".join(sys.argv[1:]) or "Wazuh: Web server 400 error code x7000 from same source ip"
    behaviour, candidates, answer = explain(alert)
    print(f"Alert:      {alert}\nBehaviour:  {behaviour}\n\nRetrieved:")
    for c in candidates:
        print(f"  {c['id']:10} {c['name']}  dist={c['distance']}")
    print(f"\nFRIDAY:     {answer}")
