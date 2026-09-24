from __future__ import annotations

import abc
import json
import logging

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import Settings

logger = logging.getLogger(__name__)


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[len("json"):]
    return text.strip()


class LLMClient(abc.ABC):
    @abc.abstractmethod
    async def complete_json(self, system: str, user: str, max_tokens: int) -> dict:
        """Call the model and return a parsed JSON dict. Raises on invalid JSON
        after retries so the caller can decide how to degrade."""
        raise NotImplementedError


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str, temperature: float = 0.1):
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def complete_json(self, system: str, user: str, max_tokens: int = 800) -> dict:
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return json.loads(_strip_json_fences(text))


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str, temperature: float = 0.1):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def complete_json(self, system: str, user: str, max_tokens: int = 800) -> dict:
        response = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = response.choices[0].message.content
        return json.loads(_strip_json_fences(text))


class HFLocalClient(LLMClient):
    """Runs an open-weights instruct model locally via transformers.

    Loaded lazily (and once) since it's the slow/heavy path — importing
    torch/transformers at module import time would make the whole codebase
    slow to import even when using an API backend.
    """

    def __init__(self, model_name: str, device: str = "cpu", hf_token: str | None = None,
                 temperature: float = 0.1):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        self.device = device
        self.temperature = max(temperature, 0.01)  # HF sampling needs > 0

    async def complete_json(self, system: str, user: str, max_tokens: int = 800) -> dict:
        # transformers generation is sync/blocking; run it in a thread so it
        # doesn't stall the FastAPI event loop.
        import asyncio

        return await asyncio.to_thread(self._generate_sync, system, user, max_tokens)

    def _generate_sync(self, system: str, user: str, max_tokens: int) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return json.loads(_strip_json_fences(text))


def build_llm_client(settings: Settings) -> LLMClient:
    backend = settings.llm_backend
    if backend == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("llm_backend='anthropic' requires ANTHROPIC_API_KEY")
        return AnthropicClient(settings.anthropic_api_key, settings.anthropic_model,
                                settings.generation_temperature)
    if backend == "openai":
        if not settings.openai_api_key:
            raise ValueError("llm_backend='openai' requires OPENAI_API_KEY")
        return OpenAIClient(settings.openai_api_key, settings.openai_model,
                             settings.generation_temperature)
    if backend == "hf_local":
        return HFLocalClient(
            settings.hf_generation_model, settings.device, settings.hf_token,
            settings.generation_temperature,
        )
    raise ValueError(f"Unknown llm_backend: {backend!r}")
