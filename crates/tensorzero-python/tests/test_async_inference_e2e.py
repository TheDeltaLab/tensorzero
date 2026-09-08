# Modified by Delta-AI under Apache 2.0
"""Live-gateway e2e tests for the async inference client methods and the
`status` / `health` endpoints.

Gated on the `TZ_E2E_KEY` and `TZ_E2E_GATEWAY` environment variables; every
test skips when either is unset. The API key must only ever be provided via
the environment, never committed to the repository.
"""

import asyncio
import inspect
import json
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List

import pytest
import pytest_asyncio
from tensorzero import (
    AsyncTaskStreamEvent,
    AsyncTensorZeroGateway,
    TensorZeroError,
    TensorZeroGateway,
)

E2E_KEY = os.environ.get("TZ_E2E_KEY")
E2E_GATEWAY = os.environ.get("TZ_E2E_GATEWAY")

pytestmark = pytest.mark.skipif(
    not E2E_KEY or not E2E_GATEWAY,
    reason="TZ_E2E_KEY / TZ_E2E_GATEWAY are not set",
)

TEST_MODEL = "deepseek-v4-flash"
CHAT_BODY: Dict[str, Any] = {
    "model": TEST_MODEL,
    "messages": [{"role": "user", "content": "Say hello in one word."}],
}

# Keep the total wait budget under pytest's 60s per-test timeout.
WAIT_KWARGS = {"initial_interval_ms": 500, "max_interval_ms": 2000, "timeout_ms": 45000}


@pytest.fixture()
def sync_client() -> Iterator[TensorZeroGateway]:
    assert E2E_KEY is not None and E2E_GATEWAY is not None
    with TensorZeroGateway.build_http(gateway_url=E2E_GATEWAY, api_key=E2E_KEY) as client:
        yield client


@pytest_asyncio.fixture()
async def async_client() -> AsyncIterator[AsyncTensorZeroGateway]:
    assert E2E_KEY is not None and E2E_GATEWAY is not None
    client_fut = AsyncTensorZeroGateway.build_http(gateway_url=E2E_GATEWAY, api_key=E2E_KEY)
    assert inspect.isawaitable(client_fut)
    client = await client_fut
    yield client
    await client.close()


def test_status_and_health(sync_client: TensorZeroGateway):
    status = sync_client.status()
    assert status["status"] == "ok"
    assert status["version"]
    health = sync_client.health()
    assert health["gateway"] == "ok"


def test_submit_chat_and_wait_completed(sync_client: TensorZeroGateway):
    launch = sync_client.submit_async_inference(kind="chat", request=CHAT_BODY)
    task_id = launch["task_id"]
    uuid.UUID(task_id)  # should be a valid UUID string

    status = sync_client.wait_for_async_task(task_id=task_id, **WAIT_KWARGS)  # type: ignore[arg-type]
    assert status["status"] == "completed"
    assert status["task_id"] == task_id
    response = status["response"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert response["object"] == "chat.completion"
    content = response["choices"][0]["message"]["content"]
    assert isinstance(content, str) and content


def test_poll_observes_valid_statuses(sync_client: TensorZeroGateway):
    launch = sync_client.submit_async_inference(kind="chat", request=CHAT_BODY)

    seen: List[str] = []
    for _ in range(150):
        status = sync_client.get_async_task(task_id=launch["task_id"])
        seen.append(status["status"])
        if status["status"] in ("completed", "failed", "cancelled"):
            break
    assert seen, "should have polled at least once"
    for name in seen[:-1]:
        assert name in ("queued", "running"), f"unexpected intermediate status: {seen}"
    assert seen[-1] == "completed", f"final status should be completed: {seen}"


def _collect_stream(
    client: TensorZeroGateway, task_id: str
) -> List[AsyncTaskStreamEvent]:
    # Attach immediately; the endpoint replays already-written events, so this
    # works whether the task is still running or just finished. Retry the
    # attach on two transient conditions observed against the shared dev
    # gateway: the gateway answers 500 when its Valkey read of the event
    # stream exceeds the command timeout, and a stream that ends quietly with
    # zero events means the connection was dropped while the task was still
    # queued (e.g. by an ingress idle timeout).
    events: List[AsyncTaskStreamEvent] = []
    for attempt in range(6):
        try:
            events = list(client.stream_async_task(task_id=task_id))
        except TensorZeroError:
            pass
        if events:
            break
        time.sleep(min(attempt + 1, 4))
    return events


def test_stream_replays_chunks_and_ends_quietly(sync_client: TensorZeroGateway):
    launch = sync_client.submit_async_inference(kind="chat", request=CHAT_BODY)

    events = _collect_stream(sync_client, launch["task_id"])
    assert events, "stream should yield at least one chunk event"

    content = ""
    for event in events:
        assert set(event.keys()) == {"event", "data"}
        assert event["data"] != "[DONE]", "[DONE] should end the stream without being yielded"
        chunk = json.loads(event["data"])
        assert chunk["object"] == "chat.completion.chunk", f"bad chunk shape: {chunk}"
        delta = chunk["choices"][0]["delta"].get("content")
        if delta:
            content += delta
    assert content

    # The status endpoint should agree that the task completed, and its final
    # response should carry the same content the stream produced.
    status = sync_client.wait_for_async_task(task_id=launch["task_id"], **WAIT_KWARGS)  # type: ignore[arg-type]
    assert status["status"] == "completed"
    response = status["response"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert response["choices"][0]["message"]["content"] == content


def test_unknown_task_404_and_bad_model_fails(sync_client: TensorZeroGateway):
    with pytest.raises(TensorZeroError) as exc_info:
        sync_client.get_async_task(task_id=str(uuid.uuid4()))
    assert "404" in str(exc_info.value)

    launch = sync_client.submit_async_inference(
        kind="chat", request={**CHAT_BODY, "model": "gpt-4o"}
    )
    status = sync_client.wait_for_async_task(task_id=launch["task_id"], **WAIT_KWARGS)  # type: ignore[arg-type]
    assert status["status"] == "failed"
    error = status["error"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert "not found in model table" in json.dumps(error)


@pytest.mark.asyncio
async def test_async_client_e2e(async_client: AsyncTensorZeroGateway):
    status = await async_client.status()
    assert status["status"] == "ok"
    health = await async_client.health()
    assert health["gateway"] == "ok"

    launch = await async_client.submit_async_inference(kind="chat", request=CHAT_BODY)
    task_id = launch["task_id"]

    # Attach to the stream before the task finishes; retry on transient open
    # failures or quiet empty streams (see `_collect_stream`).
    events = []
    for attempt in range(6):
        try:
            events = [
                event async for event in await async_client.stream_async_task(task_id=task_id)
            ]
        except TensorZeroError:
            pass
        if events:
            break
        await asyncio.sleep(min(attempt + 1, 4))
    assert events
    for event in events:
        assert event["data"] != "[DONE]"
        assert json.loads(event["data"])["object"] == "chat.completion.chunk"

    final = await async_client.wait_for_async_task(task_id=task_id, **WAIT_KWARGS)  # type: ignore[arg-type]
    assert final["status"] == "completed"
    response = final["response"]  # pyright: ignore[reportTypedDictNotRequiredAccess]
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"]["content"]
