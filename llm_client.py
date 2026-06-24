from __future__ import annotations

import json
import os
import re
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Union

import httpx


JsonDict = Dict[str, Any]
StreamCallback = Callable[[str, Any], None]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return repr(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-._")
    return cleaned or "step"


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _redact_headers(headers: Mapping[str, str]) -> JsonDict:
    sensitive = {"authorization", "proxy-authorization", "cookie", "set-cookie"}
    return {
        key: "***" if key.lower() in sensitive else value
        for key, value in headers.items()
    }


def _extract_role_chunks(
    payload: Any,
    *,
    default_role: Optional[str] = None,
) -> List[tuple]:
    if not isinstance(payload, dict):
        return []

    chunks: List[tuple] = []

    # V2 response.
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or default_role
        for item in message.get("content") or []:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append((role, item["text"]))

    # Some SSE implementations send one message directly as the event payload.
    if isinstance(payload.get("content"), list):
        role = payload.get("role") or default_role
        for item in payload["content"]:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append((role, item["text"]))

    # V1-compatible response and stream deltas.
    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        container = choice.get("message") or choice.get("delta") or {}
        role = (
            container.get("role")
            if isinstance(container, dict)
            else None
        ) or default_role or "assistant"
        content = container.get("content") if isinstance(container, dict) else None
        if isinstance(content, str):
            chunks.append((role, content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append((role, item["text"]))

    return chunks


def _copy_message(message: Mapping[str, Any]) -> JsonDict:
    copied: JsonDict = {}
    for key in (
        "role",
        "content",
        "message_id",
        "tools_state_id",
        "functions_state_id",
        "function_call",
    ):
        if key in message:
            copied[key] = message[key]
    return copied


def _extract_messages(payload: Any) -> List[JsonDict]:
    if not isinstance(payload, dict):
        return []

    messages: List[JsonDict] = []

    for message in payload.get("messages") or []:
        if isinstance(message, dict) and isinstance(message.get("role"), str):
            messages.append(_copy_message(message))

    if isinstance(payload.get("role"), str):
        messages.append(_copy_message(payload))

    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict) and isinstance(message.get("role"), str):
            messages.append(_copy_message(message))

    return messages


def _text_chunks_from_messages(messages: Iterable[Mapping[str, Any]]) -> List[tuple]:
    chunks: List[tuple] = []
    for message in messages:
        role = message.get("role") or "assistant"
        content = message.get("content")
        if isinstance(content, str):
            chunks.append((role, content))
        elif isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("text"), str):
                    chunks.append((role, item["text"]))
                function_result = item.get("function_result")
                if isinstance(function_result, dict):
                    result = function_result.get("result")
                    if isinstance(result, str):
                        chunks.append((role, result))
        function_call = message.get("function_call")
        if isinstance(function_call, dict):
            chunks.append(("function_call", _json_text(function_call)))
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("function_call"), dict):
                    chunks.append(("function_call", _json_text(item["function_call"])))
    return chunks


def message_texts_by_role(messages: Iterable[Mapping[str, Any]]) -> JsonDict:
    """Collect text-like message content by role."""
    result: JsonDict = {}
    for role, text in _text_chunks_from_messages(messages):
        resolved_role = role or "assistant"
        result[resolved_role] = result.get(resolved_role, "") + text
    return result


def _function_call_record(
    function_call: Any,
    *,
    role: Optional[str] = None,
    message: Optional[Mapping[str, Any]] = None,
) -> Optional[JsonDict]:
    if not isinstance(function_call, dict):
        return None
    record: JsonDict = {
        "role": role or "assistant",
        "function_call": dict(function_call),
    }
    if message:
        for key in ("message_id", "tools_state_id", "functions_state_id"):
            if key in message:
                record[key] = message[key]
    return record


def _extract_function_calls(
    payload: Any,
    *,
    default_role: Optional[str] = None,
) -> List[JsonDict]:
    if not isinstance(payload, dict):
        return []

    calls: List[JsonDict] = []

    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role") or default_role or "assistant"
        direct_call = _function_call_record(
            message.get("function_call"),
            role=role,
            message=message,
        )
        if direct_call:
            calls.append(direct_call)
        for item in message.get("content") or []:
            if not isinstance(item, dict):
                continue
            item_call = _function_call_record(
                item.get("function_call"),
                role=role,
                message=message,
            )
            if item_call:
                calls.append(item_call)

    if isinstance(payload.get("content"), list):
        role = payload.get("role") or default_role or "assistant"
        for item in payload["content"]:
            if not isinstance(item, dict):
                continue
            item_call = _function_call_record(
                item.get("function_call"),
                role=role,
                message=payload,
            )
            if item_call:
                calls.append(item_call)

    for choice in payload.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        container = choice.get("message") or choice.get("delta") or {}
        if not isinstance(container, dict):
            continue
        role = container.get("role") or default_role or "assistant"
        direct_call = _function_call_record(
            container.get("function_call"),
            role=role,
            message=container,
        )
        if direct_call:
            calls.append(direct_call)
        content = container.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                item_call = _function_call_record(
                    item.get("function_call"),
                    role=role,
                    message=container,
                )
                if item_call:
                    calls.append(item_call)

    return calls


def _function_call_text(calls: List[JsonDict]) -> str:
    return "\n".join(_json_text(call["function_call"]) for call in calls)


def _text_for_role(payload: Any, role: str) -> str:
    return "".join(
        text
        for chunk_role, text in _extract_role_chunks(payload)
        if chunk_role == role
    )


def _stream_texts_by_role(events: List[JsonDict]) -> JsonDict:
    result: JsonDict = {}
    active_role: Optional[str] = None
    for event in events:
        if event.get("event") != "response.message.delta":
            continue
        chunks = _extract_role_chunks(
            event.get("data"),
            default_role=active_role,
        )
        for role, text in chunks:
            resolved_role = role or active_role or "assistant"
            active_role = resolved_role
            result[resolved_role] = result.get(resolved_role, "") + text
        calls = _extract_function_calls(
            event.get("data"),
            default_role=active_role,
        )
        if calls:
            result["function_call"] = (
                result.get("function_call", "")
                + ("\n" if result.get("function_call") else "")
                + _function_call_text(calls)
            )
    return result


def _stream_function_calls(events: List[JsonDict]) -> List[JsonDict]:
    calls: List[JsonDict] = []
    active_role: Optional[str] = None
    for event in events:
        if event.get("event") != "response.message.delta":
            continue
        data = event.get("data")
        chunks = _extract_role_chunks(data, default_role=active_role)
        for role, _text in chunks:
            active_role = role or active_role or "assistant"
        calls.extend(_extract_function_calls(data, default_role=active_role))
    return calls


def _stream_messages(events: List[JsonDict]) -> List[JsonDict]:
    messages: List[JsonDict] = []
    for role, text in _stream_texts_by_role(events).items():
        if role == "function_call":
            continue
        messages.append({"role": role, "content": [{"text": text}]})
    for call in _stream_function_calls(events):
        message: JsonDict = {
            "role": call.get("role") or "assistant",
            "content": [{"function_call": call["function_call"]}],
        }
        for key in ("message_id", "tools_state_id", "functions_state_id"):
            if key in call:
                message[key] = call[key]
        messages.append(message)
    return messages


class JupyterStreamPrinter:
    """Print V2 stream deltas immediately in a Jupyter cell."""

    def __init__(self, *, show_reasoning: bool = True) -> None:
        self.show_reasoning = show_reasoning
        self._active_role: Optional[str] = None
        self._shown_roles: set = set()

    def __call__(self, event: str, data: Any) -> None:
        if event != "response.message.delta":
            return
        chunks = _extract_role_chunks(data, default_role=self._active_role)
        for role, text in chunks:
            resolved_role = role or self._active_role or "assistant"
            self._active_role = resolved_role
            if resolved_role == "reasoning" and not self.show_reasoning:
                continue
            if resolved_role not in self._shown_roles:
                if self._shown_roles:
                    print()
                print(f"[{resolved_role}]")
                self._shown_roles.add(resolved_role)
            print(text, end="", flush=True)
        calls = _extract_function_calls(data, default_role=self._active_role)
        for call in calls:
            if "function_call" not in self._shown_roles:
                if self._shown_roles:
                    print()
                print("[function_call]")
                self._shown_roles.add("function_call")
            print(_json_text(call["function_call"]), flush=True)


@dataclass
class RunResult:
    name: str
    run_dir: Path
    request: JsonDict
    response: Any
    status_code: int
    elapsed_seconds: float
    headers: JsonDict = field(default_factory=dict)
    events: List[JsonDict] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Final assistant text. Kept as the main notebook convenience property."""
        return self.assistant_text

    @property
    def texts_by_role(self) -> JsonDict:
        if self.events:
            streamed = _stream_texts_by_role(self.events)
            if streamed:
                return streamed
        result: JsonDict = {}
        for role, text in _extract_role_chunks(self.response):
            resolved_role = role or "assistant"
            result[resolved_role] = result.get(resolved_role, "") + text
        calls = self.function_calls
        if calls:
            result["function_call"] = _function_call_text(calls)
        return result

    @property
    def request_messages(self) -> List[JsonDict]:
        payload = self.request.get("json")
        if not isinstance(payload, dict):
            return []
        messages = payload.get("messages")
        return [dict(message) for message in messages if isinstance(message, dict)] if isinstance(messages, list) else []

    @property
    def response_messages(self) -> List[JsonDict]:
        if self.events:
            streamed = _stream_messages(self.events)
            if streamed:
                return streamed
        return _extract_messages(self.response)

    @property
    def history_messages(self) -> List[JsonDict]:
        return self.request_messages + self.response_messages

    @property
    def request_texts_by_role(self) -> JsonDict:
        return message_texts_by_role(self.request_messages)

    @property
    def response_texts_by_role(self) -> JsonDict:
        return self.texts_by_role

    @property
    def history_texts_by_role(self) -> JsonDict:
        return message_texts_by_role(self.history_messages)

    @property
    def assistant_text(self) -> str:
        return self.texts_by_role.get("assistant", "")

    @property
    def reasoning_text(self) -> str:
        return self.texts_by_role.get("reasoning", "")

    @property
    def function_calls(self) -> List[JsonDict]:
        if self.events:
            streamed = _stream_function_calls(self.events)
            if streamed:
                return streamed
        return _extract_function_calls(self.response)

    @property
    def function_call_text(self) -> str:
        return self.texts_by_role.get("function_call", "")

    @property
    def usage(self) -> JsonDict:
        if isinstance(self.response, dict) and isinstance(self.response.get("usage"), dict):
            return self.response["usage"]
        for event in reversed(self.events):
            data = event.get("data")
            if isinstance(data, dict) and isinstance(data.get("usage"), dict):
                return data["usage"]
        return {}

    @property
    def execution_steps(self) -> List[JsonDict]:
        if not isinstance(self.response, dict):
            return []
        additional_data = self.response.get("additional_data") or {}
        steps = additional_data.get("execution_steps") or {}
        return steps if isinstance(steps, list) else []

    def __repr__(self) -> str:
        return (
            f"RunResult(name={self.name!r}, status_code={self.status_code}, "
            f"elapsed_seconds={self.elapsed_seconds:.3f}, run_dir={str(self.run_dir)!r})"
        )


@dataclass
class PlannedStep:
    name: str
    payload: JsonDict

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlannedStep":
        name = data.get("name")
        payload = data.get("payload")
        if not isinstance(name, str) or not name:
            raise ValueError("Planned step must have a non-empty string name")
        if not isinstance(payload, dict):
            raise ValueError(f"Planned step {name!r} must have a dict payload")
        return cls(name=name, payload=dict(payload))

    def to_dict(self) -> JsonDict:
        return {
            "name": self.name,
            "payload": self.payload,
        }


@dataclass
class LLMPlan:
    """A saved sequence of chat steps that can be executed later."""

    steps: List[PlannedStep] = field(default_factory=list)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LLMPlan":
        data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        steps = data.get("steps") if isinstance(data, dict) else None
        if not isinstance(steps, list):
            raise ValueError("Plan file must contain a 'steps' list")
        return cls([PlannedStep.from_dict(step) for step in steps])

    @property
    def names(self) -> List[str]:
        return [step.name for step in self.steps]

    def add(
        self,
        name: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        replace: bool = False,
        **parameters: Any,
    ) -> "LLMPlan":
        body = dict(payload or {})
        body.update(parameters)
        step = PlannedStep(name=name, payload=body)
        existing_index = self._index_of(name)
        if existing_index is not None:
            if not replace:
                raise ValueError(f"Step {name!r} already exists")
            self.steps[existing_index] = step
        else:
            self.steps.append(step)
        return self

    def get(self, name: str) -> PlannedStep:
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(name)

    def save(self, path: Union[str, Path]) -> Path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(
            target,
            {
                "version": 1,
                "steps": [step.to_dict() for step in self.steps],
            },
        )
        return target

    def run_one(
        self,
        client: "LLMClient",
        name: str,
        *,
        on_event: Optional[StreamCallback] = None,
        **overrides: Any,
    ) -> RunResult:
        payload = dict(self.get(name).payload)
        payload.update(overrides)
        return client.step(name, payload=payload, on_event=on_event)

    def run(
        self,
        client: "LLMClient",
        sequence: Optional[Iterable[str]] = None,
        *,
        on_event: Optional[StreamCallback] = None,
        **overrides: Any,
    ) -> Dict[str, RunResult]:
        names = list(sequence) if sequence is not None else self.names
        results: Dict[str, RunResult] = {}
        for name in names:
            results[name] = self.run_one(
                client,
                name,
                on_event=on_event,
                **overrides,
            )
        return results

    def _index_of(self, name: str) -> Optional[int]:
        for index, step in enumerate(self.steps):
            if step.name == name:
                return index
        return None


class LLMRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        run_dir: Path,
        status_code: Optional[int] = None,
        response: Any = None,
    ) -> None:
        super().__init__(message)
        self.run_dir = run_dir
        self.status_code = status_code
        self.response = response


class LLMClient:
    """Thin synchronous client for notebook LLM experiments."""

    def __init__(
        self,
        *,
        base_url: str,
        cert: Union[str, Path],
        key: Union[str, Path],
        runs_dir: Union[str, Path] = "runs",
        chat_path: str = "/v2/chat/completions",
        timeout: float = 120.0,
        default_headers: Optional[Mapping[str, str]] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cert = Path(cert).expanduser()
        self.key = Path(key).expanduser()
        self.runs_dir = Path(runs_dir).expanduser()
        self.chat_path = chat_path

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        if transport is None:
            ssl_context.load_cert_chain(certfile=self.cert, keyfile=self.key)

        headers = {"Accept": "application/json"}
        if default_headers:
            headers.update(default_headers)

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            verify=ssl_context,
            transport=transport,
        )

    @classmethod
    def from_env(
        cls,
        *,
        prefix: str = "LLM_",
        runs_dir: Union[str, Path] = "runs",
        **kwargs: Any,
    ) -> "LLMClient":
        names = {
            "base_url": f"{prefix}BASE_URL",
            "cert": f"{prefix}CERT",
            "key": f"{prefix}KEY",
        }
        missing = [env_name for env_name in names.values() if not os.getenv(env_name)]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        return cls(
            base_url=os.environ[names["base_url"]],
            cert=os.environ[names["cert"]],
            key=os.environ[names["key"]],
            runs_dir=runs_dir,
            **kwargs,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def models(self, *, name: str = "models") -> RunResult:
        return self.request("GET", "/models", name=name)

    def embeddings(
        self,
        *,
        name: str = "embeddings",
        payload: Optional[Mapping[str, Any]] = None,
        **parameters: Any,
    ) -> RunResult:
        body = dict(payload or {})
        body.update(parameters)
        return self.request("POST", "/embeddings", name=name, json=body)

    def chat(
        self,
        *,
        name: str = "chat",
        payload: Optional[Mapping[str, Any]] = None,
        on_event: Optional[StreamCallback] = None,
        **parameters: Any,
    ) -> RunResult:
        body = dict(payload or {})
        body.update(parameters)
        if body.get("stream"):
            return self._stream_request(
                "POST",
                self.chat_path,
                name=name,
                json_body=body,
                on_event=on_event,
            )
        return self.request("POST", self.chat_path, name=name, json=body)

    def step(self, name: str, **parameters: Any) -> RunResult:
        return self.chat(name=name, **parameters)

    def request(
        self,
        method: str,
        path: str,
        *,
        name: str = "request",
        json: Any = None,
        **request_kwargs: Any,
    ) -> RunResult:
        run_dir = self._start_run(name)
        request_record = self._request_record(method, path, json, request_kwargs)
        _write_json(run_dir / "request.json", request_record)
        started = time.perf_counter()

        try:
            response = self._client.request(
                method,
                path,
                json=json,
                **request_kwargs,
            )
            elapsed = time.perf_counter() - started
            parsed = self._parse_response(response)
            result = RunResult(
                name=name,
                run_dir=run_dir,
                request=request_record,
                response=parsed,
                status_code=response.status_code,
                elapsed_seconds=elapsed,
                headers=dict(response.headers),
            )
            self._save_result(result)
            if response.is_error:
                self._save_error(
                    run_dir,
                    RuntimeError(f"HTTP {response.status_code}"),
                    elapsed,
                    status_code=response.status_code,
                    response=parsed,
                )
                raise LLMRequestError(
                    f"LLM API returned HTTP {response.status_code}",
                    run_dir=run_dir,
                    status_code=response.status_code,
                    response=parsed,
                )
            return result
        except LLMRequestError:
            raise
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._save_error(run_dir, exc, elapsed)
            raise LLMRequestError(
                f"LLM request failed: {exc}",
                run_dir=run_dir,
            ) from exc

    def _stream_request(
        self,
        method: str,
        path: str,
        *,
        name: str,
        json_body: Any,
        on_event: Optional[StreamCallback],
    ) -> RunResult:
        run_dir = self._start_run(name)
        request_record = self._request_record(method, path, json_body, {})
        _write_json(run_dir / "request.json", request_record)
        started = time.perf_counter()
        events: List[JsonDict] = []

        try:
            with self._client.stream(method, path, json=json_body) as response:
                response.raise_for_status()
                with (run_dir / "events.jsonl").open("w", encoding="utf-8") as sink:
                    for event in self._iter_sse(response.iter_lines()):
                        events.append(event)
                        sink.write(json.dumps(event, ensure_ascii=False) + "\n")
                        sink.flush()
                        if on_event:
                            on_event(event["event"], event["data"])

                elapsed = time.perf_counter() - started
                final_response = self._final_stream_payload(events)
                result = RunResult(
                    name=name,
                    run_dir=run_dir,
                    request=request_record,
                    response=final_response,
                    status_code=response.status_code,
                    elapsed_seconds=elapsed,
                    headers=dict(response.headers),
                    events=events,
                )
                self._save_result(result)
                return result
        except httpx.HTTPStatusError as exc:
            elapsed = time.perf_counter() - started
            exc.response.read()
            parsed = self._parse_response(exc.response)
            self._save_error(
                run_dir,
                exc,
                elapsed,
                status_code=exc.response.status_code,
                response=parsed,
            )
            raise LLMRequestError(
                f"LLM API returned HTTP {exc.response.status_code}",
                run_dir=run_dir,
                status_code=exc.response.status_code,
                response=parsed,
            ) from exc
        except Exception as exc:
            elapsed = time.perf_counter() - started
            self._save_error(run_dir, exc, elapsed)
            raise LLMRequestError(
                f"LLM stream failed: {exc}",
                run_dir=run_dir,
            ) from exc

    def _start_run(self, name: str) -> Path:
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        run_dir = self.runs_dir / f"{timestamp}-{_safe_name(name)}"
        run_dir.mkdir()
        return run_dir

    def _request_record(
        self,
        method: str,
        path: str,
        json_body: Any,
        request_kwargs: Mapping[str, Any],
    ) -> JsonDict:
        headers = dict(self._client.headers)
        headers.update(request_kwargs.get("headers") or {})
        return {
            "method": method.upper(),
            "url": f"{self.base_url}/{path.lstrip('/')}",
            "headers": _redact_headers(headers),
            "json": json_body,
            "request_options": {
                key: value
                for key, value in request_kwargs.items()
                if key != "headers"
            },
        }

    @staticmethod
    def _parse_response(response: httpx.Response) -> Any:
        try:
            return response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"text": response.text}

    @staticmethod
    def _iter_sse(lines: Iterable[str]) -> Iterator[JsonDict]:
        event_name = "message"
        data_lines: List[str] = []
        for line in lines:
            if line == "":
                if data_lines:
                    yield LLMClient._make_sse_event(event_name, data_lines)
                event_name = "message"
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "event":
                event_name = value
            elif field == "data":
                data_lines.append(value)
        if data_lines:
            yield LLMClient._make_sse_event(event_name, data_lines)

    @staticmethod
    def _make_sse_event(event_name: str, data_lines: List[str]) -> JsonDict:
        raw = "\n".join(data_lines)
        if raw == "[DONE]":
            data: Any = raw
        else:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = raw
        return {"event": event_name, "data": data}

    @staticmethod
    def _final_stream_payload(events: List[JsonDict]) -> Any:
        for event in reversed(events):
            if event["event"] == "response.message.done":
                return event["data"]
        for event in reversed(events):
            if isinstance(event.get("data"), dict):
                return event["data"]
        return {}

    @staticmethod
    def _save_error(
        run_dir: Path,
        exc: Exception,
        elapsed: float,
        *,
        status_code: Optional[int] = None,
        response: Any = None,
    ) -> None:
        _write_json(
            run_dir / "error.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "status_code": status_code,
                "response": response,
                "elapsed_seconds": elapsed,
            },
        )

    @staticmethod
    def _save_result(result: RunResult) -> None:
        _write_json(result.run_dir / "response.json", result.response)
        _write_json(
            result.run_dir / "metadata.json",
            {
                "name": result.name,
                "status_code": result.status_code,
                "elapsed_seconds": result.elapsed_seconds,
                "response_headers": _redact_headers(result.headers),
                "saved_at": datetime.now(timezone.utc),
            },
        )
        if result.response_messages:
            _write_json(result.run_dir / "response_messages.json", result.response_messages)
        if result.history_messages:
            _write_json(result.run_dir / "history_messages.json", result.history_messages)
        (result.run_dir / "text.txt").write_text(result.text, encoding="utf-8")
        (result.run_dir / "assistant.txt").write_text(
            result.assistant_text,
            encoding="utf-8",
        )
        if result.reasoning_text:
            (result.run_dir / "reasoning.txt").write_text(
                result.reasoning_text,
                encoding="utf-8",
            )
        if result.function_calls:
            _write_json(result.run_dir / "function_calls.json", result.function_calls)
            (result.run_dir / "function_call.txt").write_text(
                result.function_call_text,
                encoding="utf-8",
            )
        if result.execution_steps:
            _write_json(
                result.run_dir / "execution_steps.json",
                result.execution_steps,
            )
