"""Turn AIT training windows into MLX chat data for LoRA fine-tuning.
Only uses the 6 TRAINING scenarios - the 2 test scenarios are never seen."""
import json, random, os, collections

INSTRUCTION = "Analyze the following security alert and return its severity, a summary, and a recommended action."
TEST = {"wilson", "russellmitchell"}
ACTION = {"low": "ignore", "medium": "monitor", "high": "investigate", "critical": "escalate_immediately"}
SUMMARY = {
    "low": "Alert pattern matches normal background activity on this host.",
    "medium": "Alert pattern indicates reconnaissance such as scanning or directory brute-forcing.",
    "high": "Alert pattern indicates an active attack attempt such as password cracking.",
    "critical": "Alert pattern indicates a likely compromise such as a webshell, privilege escalation or data exfiltration.",
}

rows = [r for r in json.load(open("data/ait/windows.json")) if r["scenario"] not in TEST]
random.seed(42)
low = [r for r in rows if r["severity"] == "low"]
attack = [r for r in rows if r["severity"] != "low"]
data = attack + random.sample(low, len(attack) * 2)     # keep more normal windows, like real life
random.shuffle(data)

def to_chat(r):
    answer = {"severity": r["severity"], "summary": SUMMARY[r["severity"]],
              "recommended_action": ACTION[r["severity"]]}
    return {"messages": [{"role": "user", "content": f"{INSTRUCTION}\n\n{r['text']}"},
                         {"role": "assistant", "content": json.dumps(answer)}]}

cut = int(len(data) * 0.9)
os.makedirs("data/mlx_ait", exist_ok=True)
for name, part in [("train", data[:cut]), ("valid", data[cut:])]:
    with open(f"data/mlx_ait/{name}.jsonl", "w") as f:
        for r in part:
            f.write(json.dumps(to_chat(r)) + "\n")
    print(name, len(part), collections.Counter(r["severity"] for r in part))
