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

SYSTEM_PROMPT = """Sei NaturalGit, un assistente esperto e infallibile per Git.
Il tuo compito è convertire le richieste espresse in linguaggio naturale (italiano o inglese) nel comando Git esatto ed ottimale.

REGOLE CRITICHE:
1. Rispondi ESCLUSIVAMENTE con un oggetto JSON valido, senza formattazione markdown (niente ```json), senza testo prima o dopo.
2. Il JSON deve avere esattamente questa struttura:
{
  "command": "comando git da eseguire (es: git commit -m \\"fix\\")",
  "explanation": "Spiegazione breve e chiara in italiano di cosa fa il comando.",
  "is_destructive": false,
  "confidence": 0.95
}
3. Se l'operazione include più comandi, concatenali con " && " (es. "git add -A && git commit -m \\"...\\"").
4. "is_destructive" deve essere TRUE per comandi che possono cancellare codice o sovrascrivere la storia (es: reset --hard, clean -fd, push --force, branch -D, checkout con perdita dati).
5. Tieni conto del contesto corrente del repository fornito (branch, file modificati, staging).
"""

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
    """Queries Google Gemini API."""
    api_key = config.get("gemini_api_key")
    if not api_key:
        return None
        
    model = config.get("gemini_model", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    context_str = json.dumps(ctx_data, ensure_ascii=False)
    combined = f"{SYSTEM_PROMPT}\n\nContesto Git:\n{context_str}\n\nRichiesta utente: {prompt}"
    
    payload = {
        "contents": [{"parts": [{"text": combined}]}],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw_text = body["candidates"][0]["content"]["parts"][0]["text"].strip()
            # Clean possible markdown wrapping if any
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


def query_llm(prompt: str, ctx: Optional[GitContext], config: Dict[str, Any]) -> Optional[MatchResult]:
    """
    Attempts to translate natural language using configured LLM.
    Returns MatchResult or None if no LLM succeeded.
    """
    provider = config.get("provider", "auto").lower()
    ctx_data = ctx.to_dict() if ctx else {}
    
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
