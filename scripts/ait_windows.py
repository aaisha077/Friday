import csv, glob, collections, json, random, os, sys
SEV = {'false_positive':'low','network_scans':'medium','service_scans':'medium','dirb':'medium','wpscan':'medium',
       'cracking':'high','webshell':'critical','reverse_shell':'critical','privilege_escalation':'critical',
       'service_stop':'critical','dnsteal':'critical'}
RANK = ['low','medium','high','critical']
WIN = 300  # 5-minute windows
src = sys.argv[1]
windows = collections.defaultdict(lambda: {"names": collections.Counter(), "sev": "low"})
for f in sorted(glob.glob(os.path.join(src, '*_alerts.txt'))):
    scen = os.path.basename(f).split('_')[0]
    for r in csv.DictReader(open(f)):
        key = (scen, r['host'], int(r['time']) // WIN)
        w = windows[key]
        w["names"][r['name']] += 1
        s = SEV[r['time_label']]
        if RANK.index(s) > RANK.index(w["sev"]): w["sev"] = s
rows = []
for (scen, host, t), w in windows.items():
    text = f"host={host} | 5-min window | " + "; ".join(f"{n} x{c}" for n, c in w["names"].most_common(15))
    rows.append({"scenario": scen, "text": text, "severity": w["sev"]})
print(len(rows), collections.Counter(r["severity"] for r in rows))
json.dump(rows, open(sys.argv[2], "w"))
 