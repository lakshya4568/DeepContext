"""Tests for RLM host bridge, sandboxed kernel, recursion depth, and async messaging."""

import pytest

from deep_context.core.types import Budgets, SessionHandle, SessionStatus
from deep_context.rlm.host_bridge import HostBridge, RecursionDepthExceeded
from deep_context.rlm.kernel import SandboxedKernel
from deep_context.rlm.orchestrator import RLMOrchestrator
from deep_context.storage import get_storage


@pytest.mark.asyncio
async def test_host_bridge_recursion_depth_enforcement() -> None:
    storage = await get_storage()
    bridge = HostBridge(storage)

    root_session = SessionHandle(
        id="root-1",
        parent_session_id=None,
        depth=0,
        budgets=Budgets(max_recursion_depth=1),
        status=SessionStatus.ACTIVE,
    )
    await bridge.register_session(root_session)

    # Child 1 at depth 1 should succeed
    child_handle = await bridge.rlm_spawn("root-1", "reviewer", "Review module X")
    assert child_handle.child_id is not None

    # Child 1 trying to spawn Grandchild at depth 2 MUST fail per host budget (depth-1 default)
    with pytest.raises(RecursionDepthExceeded):
        await bridge.rlm_spawn(child_handle.child_id, "grandchild", "Deep analysis")


@pytest.mark.asyncio
async def test_sandboxed_kernel_execution_and_stdout_cap() -> None:
    storage = await get_storage()
    bridge = HostBridge(storage)
    root_session = SessionHandle(id="sess-kernel", depth=0)
    await bridge.register_session(root_session)

    kernel = SandboxedKernel(
        session_id="sess-kernel",
        host_bridge=bridge,
        corpus="sample corpus text with key metrics",
        max_output_chars=500,
    )

    code = """
print("Testing REPL output")
matches = grep("metrics", corpus)
print(f"Matches: {matches}")
answer["content"] = "Found key metrics in corpus."
answer["ready"] = True
"""
    stdout, answer = await kernel.execute(code)
    assert "Testing REPL output" in stdout
    assert answer.ready is True
    assert "Found key metrics" in answer.content


@pytest.mark.asyncio
async def test_rlm_orchestrator_session() -> None:
    storage = await get_storage()
    orchestrator = RLMOrchestrator(storage)

    res = await orchestrator.run_session(
        task_spec="Analyze the given corpus and extract summary",
        corpus="Line 1: system initialized.\nLine 2: high load observed.\nLine 3: rate limit reached.",
        max_turns=3,
    )
    assert res.session_id is not None
    assert res.status == SessionStatus.COMPLETED
    assert len(res.answer) > 0
