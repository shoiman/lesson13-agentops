"""Детермінована "LLM" для unit/integration тестів без API-ключа."""
from typing import Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    script: List[AIMessage]
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kw: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw) -> ChatResult:
        msg = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, i=0, tokens=(1000, 200)):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{i}"}],
                     usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1],
                                     "total_tokens": sum(tokens)})


def answer(text, tokens=(1500, 150)):
    return AIMessage(content=text, usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1],
                                                   "total_tokens": sum(tokens)})
