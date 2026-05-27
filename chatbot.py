"""命令行 AI 聊天机器人 —— 多轮对话，支持 Anthropic / OpenAI。"""
import logging
from config import Config
from providers import build_provider


def main() -> None:
    cfg = Config.from_env()
    logging.basicConfig(level=cfg.log_level, format="%(levelname)s %(message)s")

    provider = build_provider(cfg)
    history: list[dict] = []

    print(f"[provider={cfg.provider} model={cfg.active_model}] 输入 'exit' 退出。\n")
    while True:
        try:
            user = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user})
        if len(history) > cfg.max_history_turns * 2:
            history = history[-cfg.max_history_turns * 2:]

        reply = provider.chat(history)
        history.append({"role": "assistant", "content": reply})
        print(f"AI: {reply}\n")


if __name__ == "__main__":
    main()
