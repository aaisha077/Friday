import json, os, time
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import classification_report, accuracy_score

load_dotenv()
client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1", timeout=60, max_retries=2)
MODEL = "openai/gpt-oss-120b"

PROMPT = """You are a SOC analyst. Triage this security alert.
Reply with JSON only: {"severity": "low|medium|high|critical", "recommended_action": "ignore|monitor|investigate|escalate_immediately"}

Alert: """

def load(path):
    rows = []
    for line in open(path):
        msgs = json.loads(line)["messages"]
        rows.append((msgs[0]["content"].split("\n\n", 1)[1],
                     json.loads(msgs[1]["content"])["severity"]))
    return rows

y_true, y_pred = [], []
for i, (alert, true_sev) in enumerate(load("data/mlx/test.jsonl")):
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0, response_format={"type": "json_object"},
            messages=[{"role": "user", "content": PROMPT + alert}])
        pred = json.loads(resp.choices[0].message.content or "{}")["severity"].lower()
    except Exception as e:
        print(f"{i}: error {e}"); pred = "error"
    y_true.append(true_sev); y_pred.append(pred)
    print(f"{i+1}/216  true={true_sev:8} pred={pred}")
    time.sleep(2)   # stay under Groq's free rate limit

print(f"\nGroq zero-shot accuracy: {accuracy_score(y_true, y_pred):.1%}")
print(classification_report(y_true, y_pred, digits=3, zero_division=0))