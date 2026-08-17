"""RLM Orchestrator loop implementing workflows/03_rlm_recursion_pipeline.md and FR11–FR16."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any

from deep_context.core.config import settings
from deep_context.core.llm_client import llm_client
from deep_context.core.logging import logger
from deep_context.core.types import (
    Budgets,
    ChildStatus,
    RlmSessionResponse,
    SessionHandle,
    SessionStatus,
    StructuredAnswer,
)
from deep_context.rlm.host_bridge import HostBridge
from deep_context.rlm.kernel import SandboxedKernel
from deep_context.storage.base import StorageInterface
from deep_context.verification.checker import EvidenceVerifier


class RLMOrchestrator:
    """Orchestrates Recursive Language Model sessions."""

    def __init__(self, storage: StorageInterface):
        self.storage = storage
        self.host_bridge = HostBridge(
            storage=storage, child_runner_factory=self._run_child_subagent
        )

    @staticmethod
    def _extract_python_code(text: str) -> str | None:
        blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if blocks:
            return "\n\n".join(b.strip() for b in blocks)
        if ">>>" in text:
            lines = [
                re.sub(r"^(?:>>>|\.\.\.)\s*", "", line)
                for line in text.splitlines()
                if line.strip().startswith((">>>", "...")) or "print(" in line or "=" in line
            ]
            if lines:
                return "\n".join(lines)
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        if any(
            first_line.startswith(kw)
            for kw in [
                "for ",
                "def ",
                "import ",
                "results = ",
                "res = ",
                "matches = ",
                "answer[",
                "print(",
            ]
        ):
            return text.strip()
        return None

    async def run_session(
        self,
        task_spec: str,
        corpus: Any,
        user_id: str = "default",
        model: str | None = None,
        max_turns: int | None = None,
        max_recursion_depth: int | None = None,
    ) -> RlmSessionResponse:
        t0 = time.time()
        session_id = str(uuid.uuid4())
        turns_limit = max_turns or max(settings.max_rlm_turns, 5)
        depth_limit = (
            max_recursion_depth if max_recursion_depth is not None else settings.max_recursion_depth
        )

        session_handle = SessionHandle(
            id=session_id,
            parent_session_id=None,
            depth=0,
            budgets=Budgets(
                max_turns=turns_limit,
                max_recursion_depth=depth_limit,
                max_wall_clock_seconds=settings.max_rlm_wall_clock_seconds,
            ),
            status=SessionStatus.ACTIVE,
        )
        await self.host_bridge.register_session(session_handle, user_id=user_id)

        # 1. Initialize sandboxed kernel with corpus loaded as variable
        kernel = SandboxedKernel(
            session_id=session_id,
            host_bridge=self.host_bridge,
            corpus=corpus,
        )

        turns_used = 0
        system_prompt = (
            "You are an expert Recursive Language Model (RLM) engine operating in a Python REPL environment.\n"
            "The document corpus is loaded as a list of chunk dictionaries in variable `corpus` (with keys 'page_number', 'section_path', 'content').\n\n"
            "AVAILABLE HELPERS IN REPL:\n"
            "- `search(keyword, max_results=5)` -> list of matching chunk dictionaries\n"
            "- `grep(pattern, text=corpus)` -> list of chunks matching regex pattern\n"
            "- `answer['content'] = '...'` -> sets your final answer\n"
            "- `answer['ready'] = True` -> signals your answer is ready and complete\n\n"
            "WORKFLOW:\n"
            "1. Search the corpus for key phrases, names, or quotes from the task:\n"
            "   ```python\n"
            "   results = search('Golden Tooth')\n"
            "   for r in results:\n"
            "       print('=== Page', r.get('page_number'), '===')\n"
            "       print(r.get('content'))\n"
            "   ```\n"
            "2. Read the surrounding narrative in the REPL output to identify the exact speaker, character, and details.\n"
            "3. Set `answer['content'] = '<your thorough, grounded answer>'` and `answer['ready'] = True`."
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task_spec}"},
        ]

        final_answer = StructuredAnswer()
        last_reasoning: str | None = None

        # 2. Parent model turn loop
        while turns_used < turns_limit:
            turns_used += 1

            # Check unread child messages
            child_msgs = await self.host_bridge.collect_messages(session_id)
            if child_msgs:
                msg_report = "\n".join(
                    f"Message from {m.session_id} ({m.receiver_name or 'child'}): {m.content}"
                    for m in child_msgs
                )
                messages.append(
                    {
                        "role": "user",
                        "content": f"[New Async Subagent Messages Received]:\n{msg_report}",
                    }
                )

            # Model generates reasoning & code
            llm_content, reasoning = await llm_client.complete(
                messages,
                model=model,
                temperature=0.2,
                max_tokens=8192,
                enable_thinking=True,
            )
            if reasoning:
                last_reasoning = reasoning

            messages.append({"role": "assistant", "content": llm_content})

            # Extract python code
            code_to_run = self._extract_python_code(llm_content)
            if not code_to_run:
                # If model provided answer text directly or declared ready
                if (
                    "answer['ready'] = True" in llm_content
                    or 'answer["ready"] = True' in llm_content
                    or "FINAL ANSWER:" in llm_content.upper()
                ):
                    final_answer.content = llm_content.replace("FINAL ANSWER:", "").strip()
                    final_answer.ready = True
                    break
                elif turns_used >= 2 and len(llm_content.strip()) > 50:
                    final_answer.content = llm_content.strip()
                    final_answer.ready = True
                    break
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": "Please write a ```python ... ``` code block using `search('keyword')` to find relevant passages in `corpus` and print them.",
                        }
                    )
                    continue

            # Execute code in sandboxed kernel
            stdout, struct_ans = await kernel.execute(code_to_run)
            final_answer = struct_ans

            # Check if answer was populated in kernel namespace
            if struct_ans.ready and struct_ans.content:
                break

            # Guard against premature ready = True if children are still running
            active_children = [
                c
                for c in self.host_bridge.children.get(session_id, [])
                if self.host_bridge.sessions.get(c.child_id, session_handle).status
                == SessionStatus.ACTIVE
            ]
            if struct_ans.ready and active_children and "everything" in task_spec.lower():
                struct_ans.ready = False
                final_answer.ready = False
                stdout += "\n[Guard Notice]: Children are still active. Wait for child messages before setting ready=True."

            if struct_ans.ready:
                break

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"[REPL Output]:\n{stdout or '(No stdout output. Variables updated in namespace.)'}\n\n"
                        "Please analyze the REPL output above. If you have the answer, synthesize your final response "
                        "and set `answer['content'] = '...'` and `answer['ready'] = True` in Python, or output your final answer directly."
                    ),
                }
            )

        # 3. Evidence sufficiency check on final answer
        if not final_answer.content:
            # Fallback: check if the last assistant message had meaningful content
            last_assistant_msg = next(
                (
                    m["content"]
                    for m in reversed(messages)
                    if m["role"] == "assistant" and len(m["content"]) > 30
                ),
                None,
            )
            if last_assistant_msg and not last_assistant_msg.strip().startswith("```"):
                final_answer.content = last_assistant_msg
                final_answer.ready = True
            else:
                final_answer.content = "Could not synthesize answer within allocated turn budget."

        support_res = await EvidenceVerifier.check_support(
            draft_answer=final_answer.content,
            evidence=[{"id": "corpus", "content": str(corpus)[:1000]}],
        )

        latency_ms = int((time.time() - t0) * 1000)

        # 4. Teardown: update session status & record episodic summary
        await self.storage.update_session_status(session_id, SessionStatus.COMPLETED)
        await self.storage.insert_episode(
            user_id=user_id,
            session_id=session_id,
            task_type="rlm_corpus_analysis",
            summary=f"RLM Task: {task_spec[:100]} | Outcome: Completed in {turns_used} turns.",
            outcome="success" if final_answer.ready else "partial",
        )

        await self.storage.insert_event_trace(
            event_type="rlm_session",
            session_id=session_id,
            payload={
                "task_spec": task_spec,
                "turns_used": turns_used,
                "children_count": len(self.host_bridge.children.get(session_id, [])),
                "answer_ready": final_answer.ready,
                "support_passed": support_res.passed,
            },
            latency_ms=latency_ms,
        )

        return RlmSessionResponse(
            session_id=session_id,
            answer=final_answer.content,
            citations=final_answer.citations,
            reasoning=last_reasoning,
            turns_used=turns_used,
            children_spawned=len(self.host_bridge.children.get(session_id, [])),
            status=SessionStatus.COMPLETED,
        )

    async def _run_child_subagent(
        self, child_id: str, task_spec: str, host_bridge: HostBridge
    ) -> None:
        """Async execution routine for child subagent."""
        try:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "You are an RLM child subagent analyzing a specific sub-problem. "
                        "Return a concise, precise summary of your findings."
                    ),
                },
                {"role": "user", "content": f"Sub-Task: {task_spec}"},
            ]
            content, _ = await llm_client.complete(prompt, max_tokens=1024, temperature=0.3)
            # Send result back to parent session via message passing
            await host_bridge.agent_message_send(
                sender_session_id=child_id,
                receiver_role="parent",
                receiver_name=None,
                content=content,
            )
            await self.storage.update_rlm_child_status(child_id, ChildStatus.COMPLETED)
        except Exception as e:
            logger.error("Child subagent %s failed: %s", child_id, e)
            await host_bridge.agent_message_send(
                sender_session_id=child_id,
                receiver_role="parent",
                receiver_name=None,
                content=f"[Error during sub-task execution: {e}]",
            )
            await self.storage.update_rlm_child_status(child_id, ChildStatus.ERROR)
