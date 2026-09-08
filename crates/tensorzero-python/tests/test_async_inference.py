# Modified by Delta-AI under Apache 2.0
"""Tests for the async inference client methods (`submit_async_inference`,
`get_async_task`, `wait_for_async_task`, `stream_async_task`) and the
`status` / `health` gateway endpoints, against a mock HTTP gateway."""

import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Tuple

import pytest
import pytest_asyncio
from tensorzero import (
    AsyncInferenceApiKind,
    AsyncTensorZeroGateway,
    TensorZeroError,
    TensorZeroGateway,
    TensorZeroInternalError,
)

TEST_TASK_ID = "0190f9c4-8e3a-7b3d-9c1e-2f4a5b6c7d8e"

CHAT_COMPLETIONS_BODY = {
    "model": "tensorzero::model_name::dummy::good",
    "messages": [{"role": "user", "content": "Hello, world!"}],
}


class MockGatewayHandler(BaseHTTPRequestHandler):
    # Shared mutable state: status responses served in order (the last one repeats)
    status_sequence: List[Dict[str, Any]] = []
    status_calls: int = 0
    recorded_submit_paths: List[str] = []

    @classmethod
    def reset(cls) -> None:
        cls.status_sequence = []
        cls.status_calls = 0
        cls.recorded_submit_paths = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def _send_json(self, status: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_sse(self, frames: List[Tuple[str, str]]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for event, data in frames:
            if event:
                self.wfile.write(f"event: {event}\n".encode())
            self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def do_POST(self) -> None:
        if self.path in (
            "/v1/chat/completions/async",
            "/v1/responses/async",
            "/v1/messages/async",
        ):
            MockGatewayHandler.recorded_submit_paths.append(self.path)
            self._send_json(202, {"task_id": TEST_TASK_ID})
            return
        self._send_json(404, {"error": "not found"})

    def do_GET(self) -> None:
        if self.path == "/status":
            self._send_json(
                200,
                {"status": "ok", "version": "2026.9.0", "config_hash": "abcd1234"},
            )
            return
        if self.path == "/health":
            self._send_json(
                200,
                {
                    "gateway": "ok",
                    "clickhouse": "ok",
                    "postgres": "ok",
                    "valkey": "ok",
                    "valkey_cache": "ok",
                },
            )
            return
        if self.path == f"/v1/async_tasks/{TEST_TASK_ID}":
            index = min(
                MockGatewayHandler.status_calls,
                len(MockGatewayHandler.status_sequence) - 1,
            )
            MockGatewayHandler.status_calls += 1
            self._send_json(200, MockGatewayHandler.status_sequence[index])
            return
        if self.path == f"/v1/async_tasks/{TEST_TASK_ID}/stream":
            self._send_sse(
                [
                    ("", json.dumps({"object": "chat.completion.chunk", "n": 1})),
                    ("response.created", json.dumps({"type": "response.created"})),
                    ("", "[DONE]"),
                ]
            )
            return
        if self.path.startswith("/v1/async_tasks/"):
            self._send_json(404, {"error": "task not found"})
            return
        self._send_json(404, {"error": "not found"})


@pytest.fixture()
def mock_gateway() -> Iterator[str]:
    MockGatewayHandler.reset()
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockGatewayHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture()
def sync_client(mock_gateway: str) -> Iterator[TensorZeroGateway]:
    with TensorZeroGateway.build_http(gateway_url=mock_gateway) as client:
        yield client


@pytest_asyncio.fixture()
async def async_client(
    mock_gateway: str,
) -> Any:
    client_fut = AsyncTensorZeroGateway.build_http(gateway_url=mock_gateway)
    assert inspect.isawaitable(client_fut)
    client = await client_fut
    yield client
    await client.close()


def test_submit_async_inference_posts_to_kind_path(sync_client: TensorZeroGateway):
    cases: List[Tuple[AsyncInferenceApiKind, str]] = [
        ("chat", "/v1/chat/completions/async"),
        ("responses", "/v1/responses/async"),
        ("messages", "/v1/messages/async"),
    ]
    for kind, expected_path in cases:
        launch = sync_client.submit_async_inference(kind=kind, request=CHAT_COMPLETIONS_BODY)
        assert launch["task_id"] == TEST_TASK_ID
        assert MockGatewayHandler.recorded_submit_paths[-1] == expected_path
    assert MockGatewayHandler.recorded_submit_paths == [
        "/v1/chat/completions/async",
        "/v1/responses/async",
        "/v1/messages/async",
    ]


def test_submit_async_inference_rejects_invalid_kind(sync_client: TensorZeroGateway):
    with pytest.raises(Exception, match="Invalid async inference API kind"):
        sync_client.submit_async_inference(kind="bogus", request=CHAT_COMPLETIONS_BODY)  # pyright: ignore[reportArgumentType]


def test_get_async_task_parses_status(sync_client: TensorZeroGateway):
    MockGatewayHandler.status_sequence = [
        {"status": "completed", "task_id": TEST_TASK_ID, "response": {"id": "chatcmpl-1"}}
    ]
    status = sync_client.get_async_task(task_id=TEST_TASK_ID)
    assert status["status"] == "completed"
    assert status["task_id"] == TEST_TASK_ID
    assert status["response"] == {"id": "chatcmpl-1"}  # pyright: ignore[reportTypedDictNotRequiredAccess]


def test_get_async_task_unknown_id_raises(sync_client: TensorZeroGateway):
    MockGatewayHandler.status_sequence = []
    with pytest.raises(TensorZeroError):
        sync_client.get_async_task(task_id="0190f9c4-0000-7000-8000-000000000000")


def test_wait_for_async_task_polls_until_terminal(sync_client: TensorZeroGateway):
    MockGatewayHandler.status_sequence = [
        {"status": "queued", "task_id": TEST_TASK_ID, "queue_position": 1},
        {"status": "running", "task_id": TEST_TASK_ID},
        {"status": "completed", "task_id": TEST_TASK_ID, "response": {"id": "chatcmpl-1"}},
    ]
    status = sync_client.wait_for_async_task(
        task_id=TEST_TASK_ID,
        initial_interval_ms=10,
        max_interval_ms=20,
        timeout_ms=10000,
    )
    assert status["status"] == "completed"
    assert MockGatewayHandler.status_calls == 3


def test_wait_for_async_task_times_out(sync_client: TensorZeroGateway):
    MockGatewayHandler.status_sequence = [{"status": "queued", "task_id": TEST_TASK_ID}]
    with pytest.raises(TensorZeroInternalError, match="did not reach a terminal state"):
        sync_client.wait_for_async_task(
            task_id=TEST_TASK_ID,
            initial_interval_ms=10,
            max_interval_ms=20,
            timeout_ms=100,
        )


def test_stream_async_task_yields_events_until_done(sync_client: TensorZeroGateway):
    events = list(sync_client.stream_async_task(task_id=TEST_TASK_ID))
    assert events == [
        {"event": None, "data": json.dumps({"object": "chat.completion.chunk", "n": 1})},
        {"event": "response.created", "data": json.dumps({"type": "response.created"})},
    ]


def test_status_and_health(sync_client: TensorZeroGateway):
    status = sync_client.status()
    assert status == {"status": "ok", "version": "2026.9.0", "config_hash": "abcd1234"}
    health = sync_client.health()
    assert health["gateway"] == "ok"
    assert health["clickhouse"] == "ok"


@pytest.mark.asyncio
async def test_async_client_async_inference_methods(async_client: AsyncTensorZeroGateway):
    MockGatewayHandler.status_sequence = [
        {"status": "running", "task_id": TEST_TASK_ID},
        {"status": "completed", "task_id": TEST_TASK_ID, "response": {"id": "chatcmpl-1"}},
    ]
    launch = await async_client.submit_async_inference(
        kind="chat", request=CHAT_COMPLETIONS_BODY
    )
    assert launch["task_id"] == TEST_TASK_ID

    status = await async_client.get_async_task(task_id=TEST_TASK_ID)
    assert status["status"] == "running"

    final = await async_client.wait_for_async_task(
        task_id=TEST_TASK_ID, initial_interval_ms=10, timeout_ms=10000
    )
    assert final["status"] == "completed"
    assert final["response"] == {"id": "chatcmpl-1"}  # pyright: ignore[reportTypedDictNotRequiredAccess]

    events = [
        event async for event in await async_client.stream_async_task(task_id=TEST_TASK_ID)
    ]
    assert [event["event"] for event in events] == [None, "response.created"]

    gw_status = await async_client.status()
    assert gw_status["status"] == "ok"
    health = await async_client.health()
    assert health["postgres"] == "ok"
