"""Score Groq (zero-shot) and FRIDAY's LoRA model on the same AIT test windows.
Usage:
  python scripts/ait_llm_eval.py groq
  python scripts/ait_llm_eval.py lora     (needs `python app/api.py` running in another terminal)
Saves every answer, so you can stop and rerun - it continues where it left off.
"""
import json, os, sys, time
from sklearn.metrics import classification_report

which = sys.argv[1]
test = json.load(open("data/ait/test_windows.json"))
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = f"eval/ait_{which}{'_' + TAG if TAG else ''}.jsonl"

PROMPT = """You are a SOC analyst. Below is every alert from one host in a 5-minute window, with counts.
Judge the overall severity of what is happening:
- low: normal activity / noise
- medium: reconnaissance (port scans, directory brute-forcing, vulnerability scanning)
- high: active attack attempt (password cracking, brute force)
- critical: likely compromise (webshell, reverse shell, privilege escalation, data exfiltration, stopped services)
Reply with JSON only: {"severity": "low|medium|high|critical"}

"""

if which == "groq":
    from openai import OpenAI
    from dotenv import load_dotenv
    load_dotenv()
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                    base_url="https://api.groq.com/openai/v1", timeout=60, max_retries=2)
    def predict(text):
        r = client.chat.completions.create(model="openai/gpt-oss-120b", temperature=0,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": PROMPT + text}])
        time.sleep(2)
        return json.loads(r.choices[0].message.content or "{}")["severity"].lower()
else:
    import requests
    def predict(text):
        r = requests.post("http://localhost:8000/triage", json={"alert_text": text}, timeout=120)
        return r.json()["severity"].lower()

done = {}
if os.path.exists(OUT):
    for line in open(OUT):
        d = json.loads(line); done[d["i"]] = d["pred"]

with open(OUT, "a") as f:
    for i, row in enumerate(test):
        if i in done:
            continue
        try:
            pred = predict(row["text"])
        except Exception as e:
            print(f"{i}: error {e}"); pred = "error"
        done[i] = pred
        f.write(json.dumps({"i": i, "pred": pred}) + "\n"); f.flush()
        print(f"{i+1}/{len(test)}  true={row['severity']:8} pred={pred}")

y_true = [test[i]["severity"] for i in sorted(done)]
y_pred = [done[i] for i in sorted(done)]
print(f"\n{which} on unseen AIT attacks:")
print(classification_report(y_true, y_pred, digits=3, zero_division=0))
