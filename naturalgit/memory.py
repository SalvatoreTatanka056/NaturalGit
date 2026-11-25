"""
High-Speed Local Memory, AI Response Cache, and Few-Shot Learning Engine.
Enables sub-millisecond responses and user-trained custom rules.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CACHE_DIR = Path.home() / ".naturalgit"
AI_CACHE_FILE = CACHE_DIR / "ai_cache.json"
LEARNED_RULES_FILE = CACHE_DIR / "learned_rules.json"


def normalize_text(text: str) -> str:
    """Normalizes text by lowercasing and removing punctuation/extra whitespace."""
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    return " ".join(cleaned.split())


def get_tokens(text: str) -> set:
    """Extracts non-trivial tokens from text."""
    stop_words = {"il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "di", "a", "da", "in", "con", "su", "per", "tra", "fra", "e", "ed", "o", "the", "a", "an", "of", "to", "in", "for", "on", "with"}
    tokens = set(normalize_text(text).split())
    return tokens - stop_words


def token_similarity(text1: str, text2: str) -> float:
    """Calculates Jaccard token similarity between two phrases (0.0 to 1.0)."""
    t1 = get_tokens(text1)
    t2 = get_tokens(text2)
    if not t1 or not t2:
        return 0.0
    intersection = len(t1.intersection(t2))
    union = len(t1.union(t2))
    return intersection / union if union > 0 else 0.0


class MemoryEngine:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.ai_cache: Dict[str, Any] = self._load_json(AI_CACHE_FILE)
        self.learned_rules: List[Dict[str, Any]] = self._load_json(LEARNED_RULES_FILE, default=[])

    def _load_json(self, path: Path, default: Any = None) -> Any:
        if default is None:
            default = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _save_json(self, path: Path, data: Any):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # --- AI RESPONSE CACHE ---
    def get_from_cache(self, query: str) -> Optional[Dict[str, Any]]:
        """Checks if an exact or high-confidence match exists in the local AI cache."""
        norm = normalize_text(query)
        if norm in self.ai_cache:
            entry = self.ai_cache[norm]
            return {
                "command": entry["command"],
                "explanation": entry.get("explanation", ""),
                "is_destructive": entry.get("is_destructive", False),
                "source": "Cache locale AI"
            }

        # Check fuzzy match across cached queries
        for cached_norm, entry in self.ai_cache.items():
            sim = token_similarity(norm, cached_norm)
            if sim >= 0.85:
                return {
                    "command": entry["command"],
                    "explanation": entry.get("explanation", ""),
                    "is_destructive": entry.get("is_destructive", False),
                    "source": f"Cache locale AI (somiglianza {int(sim*100)}%)"
                }

        return None

    def store_in_cache(self, query: str, command: str, explanation: str, is_destructive: bool = False):
        """Stores a successful resolution in the AI cache for instant reuse."""
        norm = normalize_text(query)
        self.ai_cache[norm] = {
            "command": command,
            "explanation": explanation,
            "is_destructive": is_destructive
        }
        self._save_json(AI_CACHE_FILE, self.ai_cache)

    # --- USER-LEARNED RULES ---
    def learn(self, phrase: str, command: str, explanation: Optional[str] = None, is_destructive: bool = False) -> bool:
        """Teaches NaturalGit a new custom natural language mapping."""
        phrase_clean = phrase.strip()
        cmd_clean = command.strip()
        expl = explanation or f"Comando personalizzato appreso per: '{phrase_clean}'"

        # Check if rule already exists and update
        for r in self.learned_rules:
            if normalize_text(r["phrase"]) == normalize_text(phrase_clean):
                r["command"] = cmd_clean
                r["explanation"] = expl
                r["is_destructive"] = is_destructive
                self._save_json(LEARNED_RULES_FILE, self.learned_rules)
                return True

        self.learned_rules.append({
            "phrase": phrase_clean,
            "command": cmd_clean,
            "explanation": expl,
            "is_destructive": is_destructive
        })
        self._save_json(LEARNED_RULES_FILE, self.learned_rules)
        return True

    def match_learned(self, query: str) -> Optional[Dict[str, Any]]:
        """Matches a query against user-taught rules."""
        norm = normalize_text(query)
        # 1. Exact normalized match
        for r in self.learned_rules:
            if normalize_text(r["phrase"]) == norm:
                return {
                    "command": r["command"],
                    "explanation": r["explanation"],
                    "is_destructive": r.get("is_destructive", False),
                    "source": "Regola personalizzata (appresa)"
                }

        # 2. Fuzzy token similarity match
        best_match = None
        highest_sim = 0.0
        for r in self.learned_rules:
            sim = token_similarity(norm, r["phrase"])
            if sim > highest_sim and sim >= 0.70:
                highest_sim = sim
                best_match = r

        if best_match:
            return {
                "command": best_match["command"],
                "explanation": best_match["explanation"],
                "is_destructive": best_match.get("is_destructive", False),
                "source": f"Regola personalizzata (appresa - {int(highest_sim*100)}% somiglianza)"
            }

        return None

    def list_learned(self) -> List[Dict[str, Any]]:
        return self.learned_rules

    def delete_learned(self, phrase: str) -> bool:
        norm = normalize_text(phrase)
        orig_len = len(self.learned_rules)
        self.learned_rules = [r for r in self.learned_rules if normalize_text(r["phrase"]) != norm]
        if len(self.learned_rules) < orig_len:
            self._save_json(LEARNED_RULES_FILE, self.learned_rules)
            return True
        return False

    def clear_cache(self):
        self.ai_cache = {}
        self._save_json(AI_CACHE_FILE, self.ai_cache)


_memory_instance: Optional[MemoryEngine] = None

def get_memory() -> MemoryEngine:
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = MemoryEngine()
    return _memory_instance
