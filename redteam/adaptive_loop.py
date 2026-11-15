import os
import sys
import json
from typing import TypedDict
from langgraph.graph import StateGraph, END
from agentdojo.attacks import load_attack
from agent_setup import pipeline, suite
from memory import save_attack, find_similar

STRATEGIES = ["direct", "ignore_previous", "system_message", "injecagent"]
MAX_ATTEMPTS = 4


class AttackState(TypedDict):
    user_id: str
    inj_id: str
    situation: str      # text we embed: user task + attacker goal
    use_memory: bool
    candidates: list    # strategies in the order we will try them
    tried: list
    current: str
    attempts: int
    worked: bool


def retrieve(state):
    order = []
    if state["use_memory"]:
        for hit in find_similar(state["situation"], n=5):
            name = hit["attack_text"]
            if name in STRATEGIES and name not in order:
                order.append(name)
    for name in STRATEGIES:
        if name not in order:
            order.append(name)
    print("  [retrieve] order:", order)
    return {"candidates": order}


def pick(state):
    for name in state["candidates"]:
        if name not in state["tried"]:
            return {"current": name}
    return {"current": ""}


def run_attack(state):
    name = state["current"]
    attack = load_attack(name, suite, pipeline)
    user_task = suite.user_tasks[state["user_id"]]
    inj_task = suite.injection_tasks[state["inj_id"]]
    injections = attack.attack(user_task, inj_task)
    try:
        utility, worked = suite.run_task_with_pipeline(pipeline, user_task, inj_task, injections)
    except Exception as e:
        print("  [run] ERROR:", str(e)[:100])
        worked = False
    print("  [run]", name, "| attack worked:", worked)
    return {"tried": state["tried"] + [name], "attempts": state["attempts"] + 1, "worked": worked}


def save(state):
    if not state["use_memory"]:
        return {}       # memory-off runs must not teach the memory
    attack_id = state["user_id"] + "-" + state["inj_id"] + "-" + state["current"]
    save_attack(attack_id, state["situation"], state["current"], state["worked"])
    return {}


def should_continue(state):
    if state["worked"]:
        return "stop"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "stop"
    if len(state["tried"]) >= len(state["candidates"]):
        return "stop"
    return "again"


graph = StateGraph(AttackState)
graph.add_node("retrieve", retrieve)
graph.add_node("pick", pick)
graph.add_node("run_attack", run_attack)
graph.add_node("save", save)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "pick")
graph.add_edge("pick", "run_attack")
graph.add_edge("run_attack", "save")
graph.add_conditional_edges("save", should_continue, {"again": "pick", "stop": END})
app = graph.compile()


def attack_pair(user_id, inj_id, use_memory):
    user_task = suite.user_tasks[user_id]
    inj_task = suite.injection_tasks[inj_id]
    start = {
        "user_id": user_id,
        "inj_id": inj_id,
        "situation": user_task.PROMPT + " | " + inj_task.GOAL,
        "use_memory": use_memory,
        "candidates": [],
        "tried": [],
        "current": "",
        "attempts": 0,
        "worked": False,
    }
    return app.invoke(start)


if __name__ == "__main__":
    use_memory = len(sys.argv) > 1 and sys.argv[1] == "memory"
    results_file = "results_memory_on.jsonl" if use_memory else "results_memory_off.jsonl"

    done = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["user_task"], r["injection_task"]))

    user_ids = ["user_task_0", "user_task_1", "user_task_2", "user_task_3", "user_task_4"]
    inj_ids = ["injection_task_0", "injection_task_1"]

    for u in user_ids:
        for i in inj_ids:
            if (u, i) in done:
                continue
            print(u, i, "| memory:", use_memory)
            final = attack_pair(u, i, use_memory)
            row = {"user_task": u, "injection_task": i, "worked": final["worked"],
                   "attempts": final["attempts"], "winning_strategy": final["current"] if final["worked"] else None}
            with open(results_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            print("  =>", row)

    rows = []
    with open(results_file) as f:
        for line in f:
            rows.append(json.loads(line))
    wins = 0
    total_attempts = 0
    for r in rows:
        if r["worked"]:
            wins += 1
        total_attempts += r["attempts"]
    print("\nmemory:", use_memory, "| N =", len(rows), "| success:", wins, "/", len(rows),
          "| avg attempts:", round(total_attempts / len(rows), 2))