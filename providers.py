"""LLM provider 适配层 —— 统一 chat(history) -> str 接口。"""
from typing import Protocol
from config import Config


class LLMProvider(Protocol):
    def chat(self, history: list[dict]) -> str: ...


class AnthropicProvider:
    def __init__(self, cfg: Config) -> None:
        from anthropic import Anthropic
        self.cfg = cfg
        self.client = Anthropic(
            api_key=cfg.anthropic_api_key,
            base_url=cfg.anthropic_base_url,
            timeout=cfg.request_timeout,
            max_retries=cfg.max_retries,
        )

    def chat(self, history: list[dict]) -> str:
        resp = self.client.messages.create(
            model=self.cfg.anthropic_model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            system=self.cfg.system_prompt,
            stop_sequences=self.cfg.stop_sequences or None,
            messages=history,
        )
        return resp.content[0].text


class OpenAIProvider:
    def __init__(self, cfg: Config) -> None:
        from openai import OpenAI
        self.cfg = cfg
        self.client = OpenAI(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            organization=cfg.openai_org_id or None,
            project=cfg.openai_project_id or None,
            timeout=cfg.request_timeout,
            max_retries=cfg.max_retries,
        )

    def chat(self, history: list[dict]) -> str:
        messages = [{"role": "system", "content": self.cfg.system_prompt}, *history]
        resp = self.client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=messages,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            frequency_penalty=self.cfg.frequency_penalty,
            presence_penalty=self.cfg.presence_penalty,
            stop=self.cfg.stop_sequences or None,
            user=self.cfg.user_id,
        )
        return resp.choices[0].message.content or ""


def build_provider(cfg: Config) -> LLMProvider:
    if cfg.provider == "openai":
        return OpenAIProvider(cfg)
    return AnthropicProvider(cfg)
