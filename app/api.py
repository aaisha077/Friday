#!/usr/bin/env python3
"""Sift API Server — High-Performance SOC Alert Triage Endpoint
FastAPI service exposing fine-tuned Llama 3.2-3B LoRA v2 model on Apple Silicon via MLX.

Run:
    python3 app/api.py
"""

import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# MLX-LM imports
from mlx_lm import load, generate

BASE_MODEL_NAME = os.getenv("BASE_MODEL", "mlx-community/Llama-3.2-3B-Instruct-4bit")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "model/adapters/v2")
MODEL_VERSION = "llama-3.2-3b-lora-v2"

INSTRUCTION = "Analyze the following security alert and return its severity, a summary, and a recommended action."

# Global model holders
ml_models = {}

def parse_json_safely(raw_text: str):
    """Robustly parse strict JSON or extract first valid JSON block from model output."""
    raw = raw_text.strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Extract block between first { and last }
    match = re.search(r"(\{.*\})", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None

def run_inference(model, tokenizer, alert_text: str, max_tokens: int = 90):
    user_prompt = f"{INSTRUCTION}\n\n{alert_text.strip()}"
    messages = [{"role": "user", "content": user_prompt}]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    t0 = time.perf_counter()
    response_text = generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=max_tokens,
        verbose=False,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    parsed = parse_json_safely(response_text)

    return parsed, response_text, latency_ms


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load Fine-Tuned Model (LoRA v2) into memory
    print(f"\n========================================================")
    print(f"Loading Sift Fine-Tuned Model: {BASE_MODEL_NAME}")
    print(f"Applying LoRA Adapter:        {ADAPTER_PATH}")
    print(f"========================================================\n")
    
    adapter_dir = Path(ADAPTER_PATH)
    if not adapter_dir.exists():
        print(f"Warning: {ADAPTER_PATH} not found, falling back to base model only.")
        tuned_model, tuned_tokenizer, *_ = load(BASE_MODEL_NAME)
    else:
        tuned_model, tuned_tokenizer, *_ = load(BASE_MODEL_NAME, adapter_path=ADAPTER_PATH)

    ml_models["tuned_model"] = tuned_model
    ml_models["tuned_tokenizer"] = tuned_tokenizer

    # Pre-warm model graph so first request is fast
    print("Pre-warming Metal GPU graph...")
    _ = generate(tuned_model, tuned_tokenizer, prompt="warmup", max_tokens=5, verbose=False)

    # Base model loader on demand for comparison
    ml_models["base_model"] = None
    ml_models["base_tokenizer"] = None

    print(f"Model pre-warmed and ready for triage requests!\n")
    yield
    # Shutdown
    ml_models.clear()


app = FastAPI(
    title="Sift — AI Security Alert Triage API",
    description="Fine-tuned LLM microservice for SOC alert triage on Apple Silicon M3.",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for Lovable frontend and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TriageRequest(BaseModel):
    alert_text: str = Field(..., description="Raw security alert or log line string")
    model_mode: Optional[str] = Field("tuned", description="'tuned' or 'compare'")
    compare_baseline: Optional[bool] = Field(False, description="Whether to include baseline comparison")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Sift Alert Triage",
        "model_version": MODEL_VERSION,
        "adapter_loaded": ADAPTER_PATH,
    }


@app.post("/triage")
async def triage_alert(req: TriageRequest):
    if not req.alert_text.strip():
        raise HTTPException(status_code=400, detail="alert_text cannot be empty.")

    tuned_model = ml_models.get("tuned_model")
    tuned_tokenizer = ml_models.get("tuned_tokenizer")

    if not tuned_model:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    # 1. Run Fine-Tuned Model
    parsed, raw_output, latency_ms = run_inference(
        tuned_model, tuned_tokenizer, req.alert_text
    )

    if not parsed:
        severity = "medium"
        summary = raw_output.strip() or "Model output was not valid JSON."
        action = "investigate"
    else:
        severity = parsed.get("severity", "medium").lower()
        summary = parsed.get("summary", "No summary provided.")
        action = parsed.get("recommended_action") or parsed.get("action") or "investigate"

    tuned_result = {
        "severity": severity,
        "action": action,
        "recommended_action": action,
        "summary": summary,
        "model_version": MODEL_VERSION,
        "latency_ms": latency_ms,
        "source": "Live MLX Local Ingestion",
    }

    # 2. Check if comparison mode is requested
    is_compare = (req.model_mode == "compare" or req.compare_baseline is True)
    if is_compare:
        if ml_models["base_model"] is None:
            print("Loading untuned base model for comparison...")
            bm, bt, *_ = load(BASE_MODEL_NAME)
            ml_models["base_model"] = bm
            ml_models["base_tokenizer"] = bt

        base_parsed, base_raw, base_latency = run_inference(
            ml_models["base_model"], ml_models["base_tokenizer"], req.alert_text, max_tokens=80
        )
        base_result = {
            "raw": base_raw,
            "severity_guess": base_parsed.get("severity", "Unstructured markdown") if base_parsed else "Unstructured markdown",
            "latency_ms": base_latency,
            "model_version": "llama-3.2-3b-instruct (base)",
            "json_compliant": base_parsed is not None,
        }
        return {
            "tuned": tuned_result,
            "base": base_result,
        }

    return tuned_result


@app.get("/", response_class=HTMLResponse)
def home():
    """Fallback UI if someone navigates directly to port 8000 in browser."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Sift — API Server Running</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-[#0B0F17] text-slate-100 min-h-screen flex items-center justify-center p-6">
        <div class="max-w-md w-full bg-slate-900/80 border border-slate-800 rounded-2xl p-6 text-center space-y-4 shadow-2xl">
            <div class="text-3xl">⚡ SIFT API</div>
            <div class="text-emerald-400 font-mono text-sm">● Model Active & Healthy (Port 8000)</div>
            <p class="text-xs text-slate-400">Launch the frontend on port 5173 with <code class="bg-black/50 px-2 py-1 rounded text-emerald-300">npm run dev</code> inside <code class="bg-black/50 px-2 py-1 rounded text-emerald-300">frontend/</code>.</p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
