"""Group related alerts into incidents (no labels used).
1. Learn what's NORMAL per host: alert types seen on the quiet first day ("baseline").
2. Keep only alerts that are rare for that host (seen < 5 times in the baseline).
3. Chain rare alerts on the same host into one incident if they're < 15 min apart.
Usage: python scripts/incidents.py data/ait/alerts_csv/russellmitchell_alerts.txt"""
import csv, sys, collections
from datetime import datetime, timezone

GAP = 15 * 60
BASELINE_HOURS = 24

def ts(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M")

def build_incidents(path):
    rows = sorted(csv.DictReader(open(path)), key=lambda r: int(r["time"]))
    start = int(rows[0]["time"])
    cutoff = start + BASELINE_HOURS * 3600
    baseline = collections.Counter((r["host"], r["name"]) for r in rows if int(r["time"]) < cutoff)
    incidents, open_inc = [], {}
    for r in rows:
        t = int(r["time"])
        if t < cutoff or baseline[(r["host"], r["name"])] >= 5:
            continue
        inc = open_inc.get(r["host"])
        if inc is None or t - inc["end"] > GAP:
            inc = {"host": r["host"], "start": t, "end": t,
                   "alerts": collections.Counter(), "labels": collections.Counter()}
            incidents.append(inc); open_inc[r["host"]] = inc
        inc["end"] = t
        inc["alerts"][r["name"]] += 1
        inc["labels"][r["time_label"]] += 1     # ground truth, only used to CHECK results
    incidents.sort(key=lambda i: -sum(i["alerts"].values()))
    return incidents, len(rows)

def incident_text(inc):
    mins = max(1, round((inc["end"] - inc["start"]) / 60))
    return (f"host={inc['host']} | incident over {mins} min | "
            + "; ".join(f"{n} x{c}" for n, c in inc["alerts"].most_common(8)))

if __name__ == "__main__":
    incidents, n_rows = build_incidents(sys.argv[1])
    print(f"{len(incidents)} incidents from {n_rows} alerts\n")
    for inc in incidents[:10]:
        truth = ", ".join(f"{k}:{v}" for k, v in inc["labels"].most_common(3))
        print(f"[{ts(inc['start'])} → {ts(inc['end'])}] {sum(inc['alerts'].values())} alerts  (truth: {truth})")
        print(f"      {incident_text(inc)[:200]}")
