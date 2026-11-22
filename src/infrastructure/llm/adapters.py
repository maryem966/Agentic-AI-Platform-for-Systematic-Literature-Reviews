"""
Adaptateurs LLM : Ollama (local) et Groq (cloud) avec bascule automatique (Fallback).
Respecte le port LLMPort et la Section 12 du Cahier des Charges.
"""
from __future__ import annotations

import json
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from ...config.settings import settings
from ...domain.ports.llm_port import LLMPort, LLMResponse

T = TypeVar("T", bound=BaseModel)


class OllamaAdapter(LLMPort):
    """Client LLM local pour Ollama (Llama 3, Mistral, etc.)."""

    def __init__(
        self,
        base_url: str = settings.ollama_base_url,
        model: str = settings.ollama_default_model,
        timeout: float = float(settings.llm_timeout_seconds),
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    async def health_check(self) -> bool:
        """Vérifie si Ollama est allumé sur le port 11434."""
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Envoie un prompt à Ollama via son API HTTP native."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(f"{self.base_url}/api/chat", json=payload)
            res.raise_for_status()
            data = res.json()

        content = data.get("message", {}).get("content", "")
        tokens_in = data.get("prompt_eval_count", 0)
        tokens_out = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            model=self.model,
            provider="ollama",
            tokens_input=tokens_in,
            tokens_output=tokens_out,
        )

    async def invoke_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: float = 0.0,
        max_repair_attempts: int = settings.llm_max_repair_attempts,
    ) -> T:
        """Garantit une sortie validée par le schéma Pydantic (avec prompt de réparation)."""
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        system_instruction = (
            f"Tu es un assistant scientifique strict. Réponds UNIQUEMENT en JSON valide "
            f"qui respecte exactement ce schéma Pydantic :\n{schema_json}\nNe mets aucun texte avant ou après le JSON."
        )

        formatted_messages = [{"role": "system", "content": system_instruction}, *messages]

        attempts = 0
        last_error = ""

        while attempts <= max_repair_attempts:
            if attempts > 0:
                # Ajout du message de réparation (Repair Prompt - Section 12.1)
                formatted_messages.append({
                    "role": "user",
                    "content": f"Le JSON précédent était invalide : {last_error}. Corrige-le et renvoie UNIQUEMENT le JSON valide.",
                })

            resp = await self.invoke(formatted_messages, temperature=temperature)
            raw_text = resp.content.strip()

            # Nettoyage des balises markdown éventuelles (```json ... ```)
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()

            try:
                parsed_dict = json.loads(raw_text)
                return response_model.model_validate(parsed_dict)
            except Exception as e:
                last_error = str(e)
                attempts += 1

        raise ValueError(f"Échec de validation Pydantic après {max_repair_attempts} tentatives : {last_error}")


class GroqAdapter(LLMPort):
    """Client cloud ultra-rapide pour Groq (Llama 3.1 8B, etc.)."""

    def __init__(
        self,
        api_key: str = settings.groq_api_key,
        model: str = settings.groq_default_model,
        timeout: float = float(settings.llm_timeout_seconds),
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.base_url = "https://api.groq.com/openai/v1"

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self.model

    async def health_check(self) -> bool:
        """Vérifie la validité de la clé Groq."""
        if not self.api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                res = await client.get(f"{self.base_url}/models", headers=headers)
                return res.status_code == 200
        except Exception:
            return False

    async def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Appel compatible OpenAI sur l'API Groq."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            res = await client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()

        choice = data["choices"][0]
        content = choice["message"]["content"]
        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            model=self.model,
            provider="groq",
            tokens_input=usage.get("prompt_tokens", 0),
            tokens_output=usage.get("completion_tokens", 0),
        )

    async def invoke_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: float = 0.0,
        max_repair_attempts: int = settings.llm_max_repair_attempts,
    ) -> T:
        """Garantit une sortie validée par Pydantic via Groq."""
        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        system_instruction = (
            f"Tu es un assistant scientifique rigoureux. Tu dois répondre UNIQUEMENT par un JSON valide "
            f"conforme à ce schéma Pydantic :\n{schema_json}"
        )
        formatted_messages = [{"role": "system", "content": system_instruction}, *messages]

        attempts = 0
        last_error = ""

        while attempts <= max_repair_attempts:
            if attempts > 0:
                formatted_messages.append({
                    "role": "user",
                    "content": f"Erreur de validation : {last_error}. Rends UNIQUEMENT le JSON valide sans commentaires.",
                })

            resp = await self.invoke(formatted_messages, temperature=temperature)
            raw_text = resp.content.strip()

            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()

            try:
                parsed_dict = json.loads(raw_text)
                return response_model.model_validate(parsed_dict)
            except Exception as e:
                last_error = str(e)
                attempts += 1

        raise ValueError(f"Échec de validation Pydantic Groq après {max_repair_attempts} tentatives : {last_error}")


class FallbackLLMAdapter(LLMPort):
    """
    Adaptateur résilient : tente le modèle principal (ex: Ollama local).
    Si le principal plante ou timeout, bascule automatiquement sur le fallback (ex: Groq cloud).
    Section 12.1 Niveau 2 du CdC.
    """

    def __init__(self, primary: LLMPort, fallback: LLMPort) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def provider_name(self) -> str:
        return f"{self.primary.provider_name}->{self.fallback.provider_name}"

    @property
    def model_name(self) -> str:
        return self.primary.model_name

    async def health_check(self) -> bool:
        return await self.primary.health_check() or await self.fallback.health_check()

    async def invoke(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        try:
            return await self.primary.invoke(messages, temperature, max_tokens)
        except Exception:
            # Bascule transparente sur le fallback
            return await self.fallback.invoke(messages, temperature, max_tokens)

    async def invoke_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        temperature: float = 0.0,
        max_repair_attempts: int = 2,
    ) -> T:
        try:
            return await self.primary.invoke_structured(messages, response_model, temperature, max_repair_attempts)
        except Exception:
            return await self.fallback.invoke_structured(messages, response_model, temperature, max_repair_attempts)
