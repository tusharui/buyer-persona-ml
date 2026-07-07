import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LLM_API_KEY, LLM_MODEL, LLM_PROVIDER, LLM_API_BASE, PERSONA_DESCRIPTIONS, BUSINESS_RECOMMENDATIONS


SYSTEM_PROMPT = """You are a senior marketing analyst AI. Given a customer persona name and its behavioral profile,
generate a rich, detailed narrative description (2-3 paragraphs) of this customer segment. Include:
1. Who they are demographically/behaviorally
2. What drives their purchasing decisions
3. What marketing channels work best for them
4. What products they prefer
5. Their lifetime value potential

Be specific, data-driven, and actionable. Write in a professional consulting tone."""


class LLMClient:
    def __init__(self):
        self.api_key = LLM_API_KEY
        self.model = LLM_MODEL
        self.provider = LLM_PROVIDER
        self.api_base = LLM_API_BASE

    def _call_chat_completion(self, prompt: str, system: str = None) -> str:
        import httpx
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system or SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
        }
        try:
            resp = httpx.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"LLM API call failed: {e}")

    def _fallback_narrative(self, persona: str, profile: dict = None) -> str:
        desc = PERSONA_DESCRIPTIONS.get(persona, "")
        recs = BUSINESS_RECOMMENDATIONS.get(persona, [])
        text = [f"## {persona}", "", desc, "", "**Recommended Strategies:**"]
        for r in recs:
            text.append(f"- {r}")
        if profile:
            text.extend(["", "**Profile Highlights:**"])
            for k, v in sorted(profile.items())[:6]:
                text.append(f"- {k}: {v}")
        return "\n".join(text)

    def generate_narrative(self, persona: str, profile: dict = None) -> dict:
        if not self.api_key:
            return {
                "persona": persona,
                "narrative": self._fallback_narrative(persona, profile),
                "model_used": "fallback_rule_based",
            }

        prompt_parts = [f"Persona: {persona}"]
        if profile:
            prompt_parts.append("Behavioral Profile:")
            for k, v in sorted(profile.items()):
                prompt_parts.append(f"  {k}: {v}")
        prompt_parts.extend([
            "",
            "Generate a detailed marketing persona narrative (2-3 paragraphs).",
        ])
        prompt = "\n".join(prompt_parts)

        try:
            narrative = self._call_chat_completion(prompt)
            return {"persona": persona, "narrative": narrative, "model_used": self.model}
        except Exception as e:
            return {
                "persona": persona,
                "narrative": self._fallback_narrative(persona, profile),
                "model_used": f"fallback ({e})",
            }


llm_client = LLMClient()
