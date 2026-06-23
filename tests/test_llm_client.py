import json
import tempfile
import unittest
from pathlib import Path

import httpx

from llm_client import LLMClient, LLMRequestError


class LLMClientTests(unittest.TestCase):
    def make_client(self, handler, runs_dir):
        return LLMClient(
            base_url="https://llm.test",
            cert="unused.crt",
            key="unused.key",
            runs_dir=runs_dir,
            transport=httpx.MockTransport(handler),
        )

    def test_step_passes_arbitrary_parameters_and_saves_artifacts(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(body["future_parameter"], {"enabled": True})
            return httpx.Response(
                200,
                json={
                    "messages": [
                        {
                            "role": "reasoning",
                            "content": [{"text": "Сначала проверю."}],
                        },
                        {
                            "role": "assistant",
                            "content": [{"text": "Готово"}],
                        }
                    ],
                    "usage": {"total_tokens": 12},
                    "additional_data": {
                        "execution_steps": [{"event_type": "model"}]
                    },
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.make_client(handler, temp_dir) as client:
                result = client.step(
                    "test step",
                    model="model",
                    messages=[],
                    future_parameter={"enabled": True},
                )

            self.assertEqual(result.text, "Готово")
            self.assertEqual(result.assistant_text, "Готово")
            self.assertEqual(result.reasoning_text, "Сначала проверю.")
            self.assertEqual(
                result.texts_by_role,
                {
                    "reasoning": "Сначала проверю.",
                    "assistant": "Готово",
                },
            )
            self.assertEqual(result.usage, {"total_tokens": 12})
            self.assertEqual(result.execution_steps, [{"event_type": "model"}])
            self.assertTrue((result.run_dir / "request.json").exists())
            self.assertTrue((result.run_dir / "response.json").exists())
            self.assertTrue((result.run_dir / "metadata.json").exists())
            self.assertEqual(
                (result.run_dir / "text.txt").read_text(encoding="utf-8"),
                "Готово",
            )
            self.assertEqual(
                (result.run_dir / "assistant.txt").read_text(encoding="utf-8"),
                "Готово",
            )
            self.assertEqual(
                (result.run_dir / "reasoning.txt").read_text(encoding="utf-8"),
                "Сначала проверю.",
            )
            self.assertTrue((result.run_dir / "execution_steps.json").exists())

    def test_stream_collects_sse_and_saves_jsonl(self):
        stream_body = "\n".join(
            [
                "event: response.message.delta",
                'data: {"choices":[{"delta":{"content":"При"}}]}',
                "",
                "event: response.message.delta",
                'data: {"choices":[{"delta":{"content":"вет"}}]}',
                "",
                "event: response.message.done",
                'data: {"finish_reason":"stop","usage":{"total_tokens":7}}',
                "",
            ]
        )

        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=stream_body,
            )

        observed = []
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.make_client(handler, temp_dir) as client:
                result = client.step(
                    "stream",
                    model="model",
                    messages=[],
                    stream=True,
                    on_event=lambda event, data: observed.append(event),
                )

            self.assertEqual(result.text, "Привет")
            self.assertEqual(result.reasoning_text, "")
            self.assertEqual(result.usage, {"total_tokens": 7})
            self.assertEqual(len(observed), 3)
            self.assertEqual(
                len(
                    (result.run_dir / "events.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                3,
            )

    def test_v2_stream_separates_reasoning_and_assistant(self):
        stream_body = "\n".join(
            [
                "event: response.message.delta",
                'data: {"messages":[{"role":"reasoning","content":[{"text":"Думаю."}]}]}',
                "",
                "event: response.message.delta",
                'data: {"messages":[{"role":"assistant","content":[{"text":"Ответ."}]}]}',
                "",
                "event: response.message.done",
                'data: {"finish_reason":"stop","usage":{"total_tokens":9}}',
                "",
            ]
        )

        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=stream_body,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.make_client(handler, temp_dir) as client:
                result = client.step(
                    "reasoning-stream",
                    model="model",
                    messages=[],
                    stream=True,
                )

            self.assertEqual(result.reasoning_text, "Думаю.")
            self.assertEqual(result.assistant_text, "Ответ.")
            self.assertEqual(result.text, "Ответ.")

    def test_http_error_is_saved_and_exposes_run_dir(self):
        def handler(request):
            return httpx.Response(422, json={"detail": "bad request"})

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.make_client(handler, temp_dir) as client:
                with self.assertRaises(LLMRequestError) as raised:
                    client.step("failure", model="model", messages=[])

            error = raised.exception
            self.assertEqual(error.status_code, 422)
            self.assertEqual(error.response, {"detail": "bad request"})
            self.assertTrue((error.run_dir / "response.json").exists())
            self.assertTrue((error.run_dir / "metadata.json").exists())
            self.assertTrue((error.run_dir / "error.json").exists())

    def test_from_env_requires_all_variables(self):
        with self.assertRaises(ValueError):
            LLMClient.from_env(prefix="MISSING_TEST_")


if __name__ == "__main__":
    unittest.main()
