"""
LLM Integration Engine for NaturalGit.
Supports:
- Ollama (Local, offline, zero API key)
- Google Gemini (Gemini Flash / Pro)
- OpenAI & Compatible APIs (Groq, DeepSeek, LM Studio, vLLM, etc.)
"""
import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional
from .git_context import GitContext
from .heuristic_engine import MatchResult

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

SYSTEM_PROMPT = """Traduttore Git rapido. Rispondi esclusivamente in formato JSON:
{"command": "comando git da eseguire", "explanation": "breve spiegazione in italiano", "is_destructive": false}
Concatena più comandi con &&. Imposta is_destructive=true se elimina dati, rami o riscrive la storia."""

def query_ollama(prompt: str, ctx_data: Dict[str, Any], config: Dict[str, Any]) -> Optional[MatchResult]:
    """Queries local Ollama endpoint."""
    url = f"{config.get('ollama_url', 'http://localhost:11434').rstrip('/')}/api/generate"
    model = config.get("ollama_model", "llama3")
    
    context_str = json.dumps(ctx_data, ensure_ascii=False)
    user_content = f"Contesto Git:\n{context_str}\n\nRichiesta utente: {prompt}"
    
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": user_content,
        "stream": False,
        "format": "json"
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_text = body.get("response", "").strip()
            parsed = json.loads(raw_text)
            return MatchResult(
                command=parsed["command"],
                explanation=parsed.get("explanation", ""),
                is_destructive=parsed.get("is_destructive", False),
                confidence=float(parsed.get("confidence", 0.9))
            )
    except Exception:
        return None


def query_gemini(prompt: str, ctx_data: Dict[str, Any], config: Dict[str, Any]) -> Optional[MatchResult]:
    """Queries Google Gemini API with ultra-low latency parameters."""
    api_key = config.get("gemini_api_key")
    if not api_key:
        return None
        
    requested_model = config.get("gemini_model", "gemini-flash-lite-latest")
    fallback_models = ["gemini-flash-lite-latest", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
    candidate_models = [requested_model] + [m for m in fallback_models if m != requested_model]

    # Minimal context to drastically reduce TTFT (Time-To-First-Token)
    branch = ctx_data.get("branch", "main")
    staged = ctx_data.get("staged_count", 0)

    try:
        from naturalgit.memory import get_memory
        few_shot = get_memory().get_few_shot_context(max_examples=5)
    except Exception:
        few_shot = ""

    user_req = f"{SYSTEM_PROMPT}{few_shot}\nBranch: {branch}, InStaging: {staged}\nRichiesta: {prompt}"
    
    payload = {
        "contents": [{"parts": [{"text": user_req}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 120,
            "temperature": 0.1
        }
    }

    for model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        try:
            raw_text = ""
            if HAS_REQUESTS:
                resp = requests.post(url, json=payload, timeout=10)
                if resp.status_code == 200:
                    body = resp.json()
                    raw_text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
                elif resp.status_code in [404, 503]:
                    continue
                else:
                    return None
            else:
                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    raw_text = body["candidates"][0]["content"]["parts"][0]["text"].strip()

            if raw_text:
                if raw_text.startswith("```"):
                    lines = raw_text.splitlines()
                    raw_text = "\n".join(lines[1:-1])
                parsed = json.loads(raw_text)
                return MatchResult(
                    command=parsed["command"],
                    explanation=parsed.get("explanation", ""),
                    is_destructive=parsed.get("is_destructive", False),
                    confidence=float(parsed.get("confidence", 0.95))
                )
        except Exception:
            continue

    return None


def query_openai(prompt: str, ctx_data: Dict[str, Any], config: Dict[str, Any]) -> Optional[MatchResult]:
    """Queries OpenAI or compatible endpoint (Groq, Together, DeepSeek, etc.)."""
    api_key = config.get("openai_api_key")
    base_url = config.get("openai_base_url", "https://api.openai.com/v1").rstrip("/")
    model = config.get("openai_model", "gpt-4o-mini")
    
    if not api_key:
        return None
        
    url = f"{base_url}/chat/completions"
    context_str = json.dumps(ctx_data, ensure_ascii=False)
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Contesto Git:\n{context_str}\n\nRichiesta utente: {prompt}"}
        ],
        "response_format": {"type": "json_object"} if "openai.com" in base_url or "groq" in base_url else None
    }
    
    if payload["response_format"] is None:
        del payload["response_format"]

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_text = body["choices"][0]["message"]["content"].strip()
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                raw_text = "\n".join(lines[1:-1])
            parsed = json.loads(raw_text)
            return MatchResult(
                command=parsed["command"],
                explanation=parsed.get("explanation", ""),
                is_destructive=parsed.get("is_destructive", False),
                confidence=float(parsed.get("confidence", 0.95))
            )
    except Exception:
        return None


def query_llmasm(prompt: str, ctx_data: Dict[str, Any], config: Dict[str, Any]) -> Optional[MatchResult]:
    """Queries local llmasm engine (.asmbin model executed via C/NumPy)."""
    import os
    import sys
    from pathlib import Path

    # Ensure llmasm is importable from project
    llmasm_pkg_dir = Path(__file__).resolve().parent.parent / "llm.asm-main" / "llmasm"
    if llmasm_pkg_dir.exists() and str(llmasm_pkg_dir) not in sys.path:
        sys.path.insert(0, str(llmasm_pkg_dir))

    try:
        from llmasm import load_model
    except ImportError:
        return None

    # Search for model path
    model_path = config.get("local_model_path")
    if not model_path or not os.path.exists(model_path):
        candidates = [
            Path.cwd() / "models" / "qwen3-0.6b-q8_0.asmbin",
            Path(__file__).resolve().parent.parent / "llm.asm-main" / "models" / "qwen3-0.6b-q8_0.asmbin",
            Path.home() / ".naturalgit" / "models" / "qwen3-0.6b-q8_0.asmbin",
        ]
        # Also check any .asmbin in current directory models/
        if (Path.cwd() / "models").exists():
            for f in (Path.cwd() / "models").glob("*.asmbin"):
                candidates.insert(0, f)
        for cand in candidates:
            if cand.exists():
                model_path = str(cand)
                break

    if not model_path or not os.path.exists(model_path):
        return None

    backend = config.get("local_backend", "auto")
    threads = int(config.get("local_threads", 4))
    try:
        model = load_model(model_path, backend=backend, threads=threads)
        branch = ctx_data.get("branch", "main")
        staged = ctx_data.get("staged_count", 0)
        formatted_prompt = f"{SYSTEM_PROMPT}\nBranch: {branch}, InStaging: {staged}\nRichiesta: {prompt}\nJSON:"
        
        raw_text = model.generate(formatted_prompt, max_tokens=100, temperature=0.1)
        if hasattr(raw_text, "text"):
            raw_text = raw_text.text
        raw_text = str(raw_text).strip()

        # Parse JSON output
        import re
        json_match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            return MatchResult(
                command=parsed["command"],
                explanation=parsed.get("explanation", f"Eseguito con motore locale llmasm ({backend})"),
                is_destructive=parsed.get("is_destructive", False),
                confidence=0.92
            )
        elif raw_text.startswith("git "):
            return MatchResult(
                command=raw_text.splitlines()[0],
                explanation=f"Comando generato dal motore locale llmasm ({backend})",
                is_destructive=False,
                confidence=0.85
            )
    except Exception:
        return None
    return None


def query_llm(prompt: str, ctx: Optional[GitContext], config: Dict[str, Any], force_local: bool = False) -> Optional[MatchResult]:
    """
    Attempts to translate natural language using configured LLM.
    Returns MatchResult or None if no LLM succeeded.
    """
    provider = "local" if force_local else config.get("provider", "auto").lower()
    ctx_data = ctx.to_dict() if ctx else {}

    # 0. Local llmasm (.asmbin / C engine)
    if provider in ["local", "llmasm"]:
        res = query_llmasm(prompt, ctx_data, config)
        if res:
            return res
        if provider in ["local", "llmasm"]:
            return None
    
    # 1. Explicit or auto Ollama
    if provider in ["ollama", "auto"]:
        res = query_ollama(prompt, ctx_data, config)
        if res:
            return res
        if provider == "ollama":
            return None

    # 2. Explicit or auto Gemini
    if provider in ["gemini", "auto"]:
        if config.get("gemini_api_key"):
            res = query_gemini(prompt, ctx_data, config)
            if res:
                return res
            if provider == "gemini":
                return None

    # 3. Explicit or auto OpenAI
    if provider in ["openai", "auto"]:
        if config.get("openai_api_key"):
            res = query_openai(prompt, ctx_data, config)
            if res:
                return res

    return None
