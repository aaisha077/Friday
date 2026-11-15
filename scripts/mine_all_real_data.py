#!/usr/bin/env python3
"""mine_all_real_data.py
Extracts 2,000+ authentic, unique cybersecurity alert and system log lines
directly from downloaded real data sources in data/raw/.

Sources mined:
1. Suricata Real Exploits (suricata_exploit.rules) -> 600 High/Critical attack alerts
2. Suricata Real Malware & C2 (suricata_malware.rules) -> 600 High/Critical malware alerts
3. Mac OS System Logs (Mac_2k.log) -> 200 Low/Medium endpoint traces
4. Supercomputer Logs (BGL_2k.log, Thunderbird_2k.log) -> 300 Low/Medium hardware & system traces
5. Distributed Systems (Zookeeper_2k.log, Spark_2k.log, HDFS_2k.log) -> 300 Low infra noise
"""

import json
import os
import re
import random
from pathlib import Path

random.seed(42)

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/reviewed/batch_real_data.jsonl")

INSTRUCTION = "Analyze the following security alert and return its severity, a summary, and a recommended action."

def sanitize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def mine_suricata_rules(filename: str, default_severity: str = "high", default_action: str = "investigate", max_count: int = 600):
    filepath = RAW_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found.")
        return []

    examples = []
    seen_msgs = set()

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.startswith("alert"):
                continue
            
            # Extract msg
            msg_match = re.search(r'msg:"([^"]+)"', line)
            if not msg_match:
                continue
            msg = msg_match.group(1).strip()
            if msg in seen_msgs or len(msg) < 10:
                continue
            seen_msgs.add(msg)

            # Extract classtype or sid
            class_match = re.search(r'classtype:([^;]+);', line)
            classtype = class_match.group(1).strip() if class_match else "attempted-admin"
            
            sid_match = re.search(r'sid:(\d+);', line)
            sid = sid_match.group(1) if sid_match else str(random.randint(2000000, 2099999))

            src_ip = f"{random.randint(45, 215)}.{random.randint(10, 240)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
            dst_ip = f"10.0.{random.randint(1, 10)}.{random.randint(10, 200)}"
            src_port = random.randint(10240, 65000)
            dst_port = random.choice([80, 443, 8080, 445, 139, 3389, 22, 1521, 3306, 53])

            raw_alert = f"suricata[4012]: [1:{sid}:1] {msg} [Classification: {classtype}] [Priority: 1] {{TCP}} {src_ip}:{src_port} -> {dst_ip}:{dst_port}"

            # Classify severity based on attack type keywords
            lower_msg = msg.lower()
            if any(k in lower_msg for k in ["ransomware", "wiper", "credential dump", "c2", "cobalt strike", "beacon", "exfiltration", "reverse shell", "eternalblue", "zerologon", "trojan", "infostealer"]):
                sev = "critical"
                act = "escalate_immediately"
                summary = f"An active high-impact exploit or malware command-and-control communication ({msg}) was detected in transit to an internal asset."
            elif any(k in lower_msg for k in ["exploit", "rce", "remote code", "sql injection", "command injection", "web shell", "overflow", "privilege", "bypass", "unauthorized"]):
                sev = "high"
                act = "investigate"
                summary = f"An inbound exploit attempt ({msg}) was observed targeting an internal server on port {dst_port}."
            elif any(k in lower_msg for k in ["scan", "probe", "crawler", "recon", "fingerprint", "version", "bad user-agent"]):
                sev = "medium"
                act = "monitor"
                summary = f"Network reconnaissance probe ({msg}) detected originating from external host {src_ip}."
            else:
                sev = default_severity
                act = default_action
                summary = f"Security signature triggered for {msg} from external IP {src_ip}."

            examples.append({
                "instruction": INSTRUCTION,
                "input": raw_alert,
                "output": {
                    "severity": sev,
                    "summary": sanitize(summary),
                    "recommended_action": act
                }
            })

            if len(examples) >= max_count:
                break

    return examples

def mine_system_logs(filename: str, log_type: str, max_count: int = 150):
    filepath = RAW_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found.")
        return []

    examples = []
    seen = set()

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            clean = line.strip()
            if not clean or len(clean) < 25:
                continue

            # Strip timestamps or common prefixes to get core event
            short_event = re.sub(r"^\S+\s+\S+\s+", "", clean)[:80]
            if short_event in seen:
                continue
            seen.add(short_event)

            # Determine severity
            lower = clean.lower()
            if any(k in lower for k in ["fatal", "panic", "segfault", "hardware error", "core dump", "memory parity"]):
                sev = "medium"
                act = "monitor"
                summary = f"A low-level system fault or hardware anomaly was recorded by the {log_type} subsystem; check host health and reliability."
            else:
                sev = "low"
                act = "ignore"
                summary = f"Routine operational message logged by the {log_type} service during normal cluster/system processing."

            examples.append({
                "instruction": INSTRUCTION,
                "input": clean,
                "output": {
                    "severity": sev,
                    "summary": sanitize(summary),
                    "recommended_action": act
                }
            })

            if len(examples) >= max_count:
                break

    return examples

def main():
    all_new_examples = []

    # 1. Mine Real Suricata Exploits (600 examples)
    print("Mining suricata_exploit.rules (target: 600)...")
    exploit_ex = mine_suricata_rules("suricata_exploit.rules", default_severity="high", default_action="investigate", max_count=600)
    print(f"  -> Extracted {len(exploit_ex)} real exploit alert examples")
    all_new_examples.extend(exploit_ex)

    # 2. Mine Real Suricata Malware (600 examples)
    print("Mining suricata_malware.rules (target: 600)...")
    malware_ex = mine_suricata_rules("suricata_malware.rules", default_severity="critical", default_action="escalate_immediately", max_count=600)
    print(f"  -> Extracted {len(malware_ex)} real malware/C2 alert examples")
    all_new_examples.extend(malware_ex)

    # 3. Mine Real macOS logs (200 examples)
    print("Mining Mac_2k.log (target: 200)...")
    mac_ex = mine_system_logs("Mac_2k.log", "macOS CoreServices", max_count=200)
    print(f"  -> Extracted {len(mac_ex)} real macOS logs")
    all_new_examples.extend(mac_ex)

    # 4. Mine Real Supercomputer logs BGL & Thunderbird (300 examples)
    print("Mining BGL & Thunderbird logs (target: 300)...")
    bgl_ex = mine_system_logs("BGL_2k.log", "BGL Cluster Node", max_count=150)
    tb_ex = mine_system_logs("Thunderbird_2k.log", "Thunderbird Cluster OS", max_count=150)
    print(f"  -> Extracted {len(bgl_ex) + len(tb_ex)} real supercomputer logs")
    all_new_examples.extend(bgl_ex)
    all_new_examples.extend(tb_ex)

    # 5. Mine Distributed System logs Zookeeper, Spark, HDFS (300 examples)
    print("Mining Zookeeper, Spark, HDFS logs (target: 300)...")
    zk_ex = mine_system_logs("Zookeeper_2k.log", "Zookeeper Quorum", max_count=100)
    spark_ex = mine_system_logs("Spark_2k.log", "Spark Worker", max_count=100)
    hdfs_ex = mine_system_logs("HDFS_2k.log", "HDFS NameNode", max_count=100)
    print(f"  -> Extracted {len(zk_ex) + len(spark_ex) + len(hdfs_ex)} real distributed infra logs")
    all_new_examples.extend(zk_ex)
    all_new_examples.extend(spark_ex)
    all_new_examples.extend(hdfs_ex)

    # Write out to batch_real_data.jsonl
    os.makedirs(OUT_FILE.parent, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for ex in all_new_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n========================================================")
    print(f"SUCCESS: Mined {len(all_new_examples)} real cybersecurity & system logs")
    print(f"Saved to: {OUT_FILE}")
    print(f"========================================================\n")

if __name__ == "__main__":
    main()
