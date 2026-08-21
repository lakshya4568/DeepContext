"""Sandboxed Python REPL Kernel implementing ARCHITECTURE.md §4–5 and FR11, FR14, FR15."""

from __future__ import annotations

import io
import re
import sys
from typing import Any

from deep_context.core.config import settings
from deep_context.core.types import StructuredAnswer
from deep_context.rlm.host_bridge import HostBridge


class SandboxedKernel:
    """
    Persistent Python REPL execution environment for RLM sessions.
    Corpus is loaded as a variable ('corpus'), not pasted into the prompt.
    Output is strictly bounded to max_repl_chars_per_turn (default 8,192).
    """

    def __init__(
        self,
        session_id: str,
        host_bridge: HostBridge,
        corpus: Any,
        max_output_chars: int | None = None,
    ):
        self.session_id = session_id
        self.host_bridge = host_bridge
        self.max_output_chars = max_output_chars or settings.max_repl_chars_per_turn
        self.namespace: dict[str, Any] = {}
        self.answer: dict[str, Any] = {"content": "", "ready": False, "citations": []}

        self._setup_namespace(corpus)

    def _setup_namespace(self, corpus: Any) -> None:
        """Inject corpus and client stubs into the kernel namespace."""
        self.namespace = {
            "corpus": corpus,
            "answer": self.answer,
            "re": re,
            # Thin client stubs forwarding to the host
            "retrieve": self._stub_retrieve,
            "memory_save_fact": self._stub_memory_save_fact,
            "rlm_spawn": self._stub_rlm_spawn,
            "agent_message_send": self._stub_agent_message_send,
            "collect_messages": self._stub_collect_messages,
            "search": self._search,
            "grep": self._grep,
        }

    def _search(self, keyword: str, max_results: int = 5) -> list[Any]:
        """Convenience keyword search across loaded corpus items."""
        kw = keyword.lower()
        corpus_data = self.namespace.get("corpus", [])
        if isinstance(corpus_data, list):
            results: list[Any] = []
            for item in corpus_data:
                if isinstance(item, dict):
                    if kw in item.get("content", "").lower():
                        results.append(item)
                elif isinstance(item, str) and kw in item.lower():
                    results.append(item)
            return results[:max_results]
        elif isinstance(corpus_data, str):
            lines = corpus_data.splitlines()
            return [line for line in lines if kw in line.lower()][:max_results]
        return []

    # Stubs
    async def _stub_retrieve(self, query: str, **kwargs: Any) -> Any:
        return await self.host_bridge.retrieve(self.session_id, query, **kwargs)

    async def _stub_memory_save_fact(self, observation_dict: dict[str, Any]) -> Any:
        return await self.host_bridge.memory_save_fact(self.session_id, observation_dict)

    async def _stub_rlm_spawn(self, name: str, task_spec: str, model: str | None = None) -> Any:
        return await self.host_bridge.rlm_spawn(self.session_id, name, task_spec, model)

    async def _stub_agent_message_send(
        self, receiver_role: str, content: str, receiver_name: str | None = None
    ) -> None:
        await self.host_bridge.agent_message_send(
            self.session_id, receiver_role, receiver_name, content
        )

    async def _stub_collect_messages(self) -> list[Any]:
        return await self.host_bridge.collect_messages(self.session_id)

    def _grep(self, pattern: str, text: Any = None) -> list[Any]:
        """Convenience regex matcher inside the REPL."""
        target = text if text is not None else self.namespace.get("corpus", [])
        regex = re.compile(pattern, re.IGNORECASE)
        if isinstance(target, str):
            lines = target.splitlines()
            return [line for line in lines if regex.search(line)]
        elif isinstance(target, list):
            res: list[Any] = []
            for item in target:
                if isinstance(item, dict):
                    if regex.search(item.get("content", "")):
                        res.append(item)
                elif isinstance(item, str) and regex.search(item):
                    res.append(item)
            return res
        return []

    async def execute(self, code: str) -> tuple[str, StructuredAnswer]:
        """
        Executes model-generated Python code in the persistent namespace.
        Captures and truncates stdout to max_output_chars.
        Returns (stdout, current_structured_answer).
        """
        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        execution_error: str | None = None

        try:
            # Wrap in async execution block if code contains await
            if "await " in code:
                # Wrap in async runner
                indented_code = "\n".join("    " + line for line in code.splitlines())
                wrapped_code = f"async def __rlm_async_exec__():\n{indented_code}\n"
                exec(wrapped_code, self.namespace)
                await self.namespace["__rlm_async_exec__"]()
            else:
                exec(code, self.namespace)
        except Exception as e:
            execution_error = f"{type(e).__name__}: {e}"
        finally:
            sys.stdout = old_stdout

        output = redirected_output.getvalue()
        if execution_error:
            output += f"\n[Execution Error]: {execution_error}"

        # Bounded REPL output (FR14: default 8,192 chars)
        if len(output) > self.max_output_chars:
            output = (
                output[: self.max_output_chars]
                + f"\n... [REPL Output truncated at {self.max_output_chars} characters. Use targeted search/filter instead of dumping raw corpus.]"
            )

        # Sync answer
        ans_dict = self.namespace.get("answer", self.answer)
        struct_ans = StructuredAnswer(
            content=ans_dict.get("content", ""),
            ready=bool(ans_dict.get("ready", False)),
            citations=ans_dict.get("citations", []),
        )

        return output, struct_ans
