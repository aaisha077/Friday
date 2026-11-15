"""FRIDAY as a LangGraph agent.
Single alert:   python scripts/friday_graph.py "host=mail | ... x400"
    alert -> triage (fine-tuned Sift model) -> if not low: investigate -> report
Incidents:      python scripts/friday_graph.py --incidents data/ait/alerts_csv/russellmitchell_alerts.txt [how_many]
    incidents are already noise-filtered, so they skip the triage gate and go straight to investigate.
Needs the Sift API running for single alerts:  python app/api.py"""
import json, re, sys
from typing import TypedDict
from langgraph.graph import StateGraph, END

class State(TypedDict, total=False):
    alert: str
    from_incident: bool
    quiet: bool
    severity: str
    action: str
    technique: str
    explanation: str

def triage(state: State) -> State:
    if state.get("from_incident"):
        return {}                                 # already filtered as unusual -> no gate
    import requests
    r = requests.post("http://localhost:8000/triage", json={"alert_text": state.get("alert", "")}, timeout=120).json()
    return {"severity": r["severity"], "action": r["recommended_action"]}

def needs_investigation(state: State) -> str:
    return "investigate" if state.get("from_incident") or state.get("severity", "") != "low" else "report"

def investigate(state: State) -> State:
    from explain_alert import explain
    _, _, answer = explain(state.get("alert", ""))
    m = re.search(r"\{.*\}", answer, re.DOTALL)
    try:
        data = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        data = {}
    return {"technique": data.get("technique_id", "none"),
            "severity": data.get("severity", state.get("severity", "unknown")),
            "explanation": data.get("explanation", answer)}

def report(state: State) -> State:
    if state.get("quiet"):
        return {}
    print(f"\nALERT:     {state.get('alert', '')[:160]}")
    print(f"SEVERITY:  {state.get('severity')}" + (f"  ->  {state.get('action', '')}" if state.get("action") else ""))
    if state.get("technique"):
        print(f"TECHNIQUE: {state.get('technique', '')}  https://attack.mitre.org/techniques/{state.get('technique', '').replace('.', '/')}")
        print(f"FRIDAY:    {state.get('explanation', '')}")
    else:
        print("FRIDAY:    Looks like normal activity, no investigation needed.")
    return {}

graph = StateGraph(State)
graph.add_node("triage", triage)
graph.add_node("investigate", investigate)
graph.add_node("report", report)
graph.set_entry_point("triage")
graph.add_conditional_edges("triage", needs_investigation, {"investigate": "investigate", "report": "report"})
graph.add_edge("investigate", "report")
graph.add_edge("report", END)
friday = graph.compile()

RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--incidents":
        from incidents import build_incidents, incident_text, ts
        top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 15
        incidents, n = build_incidents(sys.argv[2])
        print(f"{n} alerts -> {len(incidents)} incidents. Investigating the {top_n} largest...")
        results = []
        for i, inc in enumerate(incidents[:top_n], 1):
            out = friday.invoke({"alert": incident_text(inc), "from_incident": True, "quiet": True})
            truth = inc["labels"].most_common(1)[0][0]          # ground truth, only for checking
            results.append((out, inc, truth))
            print(f"  {i}/{top_n} {inc['host']:18} -> {out.get('severity')}")

        # FRIDAY's queue: most severe first, bigger incidents first within the same severity
        results.sort(key=lambda r: (RANK.get(str(r[0].get("severity")), 4), -sum(r[1]["alerts"].values())))
        print(f"\n{'#':>2}  {'severity':9} {'technique':10} {'host':18} {'when':12} truth")
        for i, (out, inc, truth) in enumerate(results, 1):
            mark = "" if truth == "false_positive" else "  <- real attack"
            print(f"{i:>2}  {str(out.get('severity')):9} {str(out.get('technique')):10} {inc['host']:18} {ts(inc['start']):12} {truth}{mark}")
        top = results[0][0]
        print(f"\nFRIDAY's #1: {top.get('explanation')}")
    else:
        alert = " ".join(sys.argv[1:]) or "host=mail | 5-min window | Wazuh: Dovecot Authentication Success. x400"
        friday.invoke({"alert": alert})
