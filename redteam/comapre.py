import json

def load(path):
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[(r["user_task"], r["injection_task"])] = r
    return rows

base = load("results_baseline.jsonl")
off = load("results_memory_off.jsonl")
on = load("results_memory_on.jsonl")

pairs = list(off.keys())
b = 0
for p in pairs:
    if base[p]["attack_worked"]:
        b += 1

def summary(rows):
    wins = 0
    attempts = 0
    for p in pairs:
        if rows[p]["worked"]:
            wins += 1
        attempts += rows[p]["attempts"]
    return wins, round(attempts / len(pairs), 2)

print("Same", len(pairs), "pairs:")
print("baseline (1 fixed attack):", b, "/", len(pairs))
print("loop, memory off:", summary(off))
print("loop, memory on: ", summary(on))