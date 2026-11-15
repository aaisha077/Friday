#!/usr/bin/env python3
"""Evaluate base or fine-tuned model on held-out test set (data/mlx/test.jsonl).

Usage:
    # Evaluate untuned base model (baseline):
    python3 scripts/evaluate.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --out eval/results_baseline.json

    # Evaluate fine-tuned model (LoRA adapter):
    python3 scripts/evaluate.py --model mlx-community/Llama-3.2-3B-Instruct-4bit --adapter-path model/adapters/v1 --out eval/results_lora_v1.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from mlx_lm import load, generate

VALID_SEVERITIES = {"low", "medium", "high", "critical"}
VALID_ACTIONS = {"ignore", "monitor", "investigate", "escalate_immediately"}


def parse_json_output(raw_text: str):
    """Attempt to parse strict JSON or extract the first JSON object from model output."""
    raw_text = raw_text.strip()
    
    # 1. Direct parse attempt
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return data, True, "exact_json"
    except Exception:
        pass

    # 2. Extract substring between first { and last }
    match = re.search(r"(\{.*\})", raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data, False, "extracted_json_with_extra_text"
        except Exception:
            pass

    return None, False, "invalid_json"


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sift model on test.jsonl")
    parser.add_argument("--model", type=str, default="mlx-community/Llama-3.2-3B-Instruct-4bit", help="Base model path or HF name")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to trained LoRA adapter directory")
    parser.add_argument("--test-data", type=str, default="data/mlx/test.jsonl", help="Path to test set JSONL")
    parser.add_argument("--out", type=str, default="eval/results_baseline.json", help="Path to save evaluation report JSON")
    parser.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate per prompt")
    args = parser.parse_args()

    test_path = Path(args.test_data)
    if not test_path.exists():
        print(f"Error: {test_path} not found. Run scripts/prepare_data.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"\n========================================================")
    print(f"Loading model: {args.model}")
    if args.adapter_path:
        print(f"Applying LoRA adapter: {args.adapter_path}")
    else:
        print(f"Mode: Baseline (Untuned Base Model)")
    print(f"========================================================\n")

    model, tokenizer, *_ = load(args.model, adapter_path=args.adapter_path)

    with open(test_path, "r", encoding="utf-8") as f:
        test_examples = [json.loads(line.strip()) for line in f if line.strip()]

    print(f"Evaluating on {len(test_examples)} test examples...\n")

    results = []
    num_valid_json = 0
    num_exact_json = 0
    severity_correct = 0
    action_correct = 0

    for idx, item in enumerate(test_examples, 1):
        messages = item["messages"]
        user_prompt = messages[0]["content"]
        ground_truth_str = messages[1]["content"]
        ground_truth = json.loads(ground_truth_str)

        # Apply chat template for prompt only
        prompt_chat = [{"role": "user", "content": user_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            prompt_chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        response_text = generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        )

        parsed, is_exact, status = parse_json_output(response_text)

        pred_sev = parsed.get("severity") if parsed else None
        pred_act = parsed.get("recommended_action") if parsed else None
        pred_summary = parsed.get("summary") if parsed else None

        # Check correctness
        gt_sev = ground_truth.get("severity")
        gt_act = ground_truth.get("recommended_action")

        sev_match = (pred_sev == gt_sev)
        act_match = (pred_act == gt_act)

        if is_exact:
            num_exact_json += 1
        if parsed is not None:
            num_valid_json += 1
        if sev_match:
            severity_correct += 1
        if act_match:
            action_correct += 1

        print(f"[{idx:02d}/{len(test_examples):02d}] GT: {gt_sev:8s} | Pred: {str(pred_sev):8s} | SevMatch: {'✓' if sev_match else '✗'} | ExactJSON: {'✓' if is_exact else '✗'}")

        results.append({
            "id": idx,
            "user_prompt": user_prompt,
            "ground_truth": ground_truth,
            "raw_response": response_text,
            "parsed_output": parsed,
            "parse_status": status,
            "is_exact_json": is_exact,
            "severity_match": sev_match,
            "action_match": act_match,
        })

    n = len(test_examples)
    summary = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "num_test_samples": n,
        "exact_json_rate": round(num_exact_json / n * 100, 2),
        "parseable_json_rate": round(num_valid_json / n * 100, 2),
        "severity_accuracy": round(severity_correct / n * 100, 2),
        "action_accuracy": round(action_correct / n * 100, 2),
    }

    print(f"\n========================================================")
    print(f"EVALUATION SUMMARY")
    print(f"========================================================")
    print(f"Total Test Samples:        {summary['num_test_samples']}")
    print(f"Exact JSON-only Output:    {num_exact_json}/{n} ({summary['exact_json_rate']}%)")
    print(f"Parseable JSON Output:     {num_valid_json}/{n} ({summary['parseable_json_rate']}%)")
    print(f"Severity Accuracy:         {severity_correct}/{n} ({summary['severity_accuracy']}%)")
    print(f"Recommended Action Acc:    {action_correct}/{n} ({summary['action_accuracy']}%)")
    print(f"========================================================\n")

    os.makedirs(Path(args.out).parent, exist_ok=True)
    report = {
        "summary": summary,
        "detailed_results": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Detailed results saved to {args.out}\n")


if __name__ == "__main__":
    main()
