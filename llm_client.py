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
    return result


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
        return result

    @property
    def assistant_text(self) -> str:
        return self.texts_by_role.get("assistant", "")

    @property
    def reasoning_text(self) -> str:
        return self.texts_by_role.get("reasoning", "")

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
        if result.execution_steps:
            _write_json(
                result.run_dir / "execution_steps.json",
                result.execution_steps,
            )
