"""加载 .env 中的配置并包装成 Config 对象。"""
from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str) -> list[str]:
    raw = os.getenv(name, "").strip()
    return [s for s in (x.strip() for x in raw.split(",")) if s]


@dataclass(frozen=True)
class Config:
    # provider 选择
    provider: str  # "anthropic" | "openai"

    # Anthropic
    anthropic_api_key: str
    anthropic_base_url: str
    anthropic_api_version: str
    anthropic_model: str
    anthropic_fallback_model: str

    # OpenAI
    openai_api_key: str
    openai_base_url: str
    openai_org_id: str
    openai_project_id: str
    openai_model: str
    openai_fallback_model: str

    # 通用采样
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    stop_sequences: list[str] = field(default_factory=list)

    # 对话
    system_prompt: str = ""
    max_history_turns: int = 20
    stream: bool = True

    # 扩展能力
    enable_thinking: bool = False
    thinking_budget_tokens: int = 2048
    enable_prompt_cache: bool = True
    enable_tool_use: bool = False

    # 网络
    request_timeout: int = 60
    max_retries: int = 3
    retry_backoff: float = 2.0
    http_proxy: str = ""

    # 运行时
    log_level: str = "INFO"
    log_file: str = ""
    user_id: str = "local-dev"

    @property
    def active_model(self) -> str:
        return self.openai_model if self.provider == "openai" else self.anthropic_model

    @classmethod
    def from_env(cls) -> "Config":
        provider = os.getenv("PROVIDER", "anthropic").strip().lower()
        if provider not in {"anthropic", "openai"}:
            raise RuntimeError(f"PROVIDER 必须是 anthropic 或 openai，当前: {provider}")

        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if provider == "anthropic" and not anthropic_key:
            raise RuntimeError("provider=anthropic 但缺少 ANTHROPIC_API_KEY")
        if provider == "openai" and not openai_key:
            raise RuntimeError("provider=openai 但缺少 OPENAI_API_KEY")

        return cls(
            provider=provider,
            anthropic_api_key=anthropic_key,
            anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            anthropic_api_version=os.getenv("ANTHROPIC_API_VERSION", "2023-06-01"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
            anthropic_fallback_model=os.getenv("ANTHROPIC_FALLBACK_MODEL", "claude-haiku-4-5-20251001"),
            openai_api_key=openai_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_org_id=os.getenv("OPENAI_ORG_ID", ""),
            openai_project_id=os.getenv("OPENAI_PROJECT_ID", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            openai_fallback_model=os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini"),
            max_tokens=int(os.getenv("MAX_TOKENS", "1024")),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            top_p=float(os.getenv("TOP_P", "1.0")),
            top_k=int(os.getenv("TOP_K", "0")),
            frequency_penalty=float(os.getenv("FREQUENCY_PENALTY", "0.0")),
            presence_penalty=float(os.getenv("PRESENCE_PENALTY", "0.0")),
            stop_sequences=_list("STOP_SEQUENCES"),
            system_prompt=os.getenv("SYSTEM_PROMPT", "你是一个友好的助手。"),
            max_history_turns=int(os.getenv("MAX_HISTORY_TURNS", "20")),
            stream=_bool("STREAM", "true"),
            enable_thinking=_bool("ENABLE_THINKING", "false"),
            thinking_budget_tokens=int(os.getenv("THINKING_BUDGET_TOKENS", "2048")),
            enable_prompt_cache=_bool("ENABLE_PROMPT_CACHE", "true"),
            enable_tool_use=_bool("ENABLE_TOOL_USE", "false"),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "60")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_backoff=float(os.getenv("RETRY_BACKOFF", "2.0")),
            http_proxy=os.getenv("HTTP_PROXY", ""),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", ""),
            user_id=os.getenv("USER_ID", "local-dev"),
        )
