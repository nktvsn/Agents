import json
import tempfile
import unittest
from pathlib import Path

import httpx

from llm_client import LLMClient, LLMPlan, LLMRequestError


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

    def test_v2_function_call_output_is_extracted_and_saved(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "messages": [
                        {
                            "role": "assistant",
                            "tools_state_id": "state-1",
                            "content": [
                                {
                                    "function_call": {
                                        "name": "weather_forecast",
                                        "arguments": "{\"location\":\"Манжерок\"}",
                                    }
                                }
                            ],
                        }
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.make_client(handler, temp_dir) as client:
                result = client.step("function-call", model="model", messages=[])

            self.assertEqual(result.assistant_text, "")
            self.assertEqual(
                result.function_calls,
                [
                    {
                        "role": "assistant",
                        "function_call": {
                            "name": "weather_forecast",
                            "arguments": "{\"location\":\"Манжерок\"}",
                        },
                        "tools_state_id": "state-1",
                    }
                ],
            )
            self.assertIn("weather_forecast", result.function_call_text)
            self.assertEqual(
                result.texts_by_role["function_call"],
                result.function_call_text,
            )
            self.assertTrue((result.run_dir / "function_calls.json").exists())
            self.assertTrue((result.run_dir / "function_call.txt").exists())

    def test_v1_function_call_output_is_extracted(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "role": "assistant",
                                "function_call": {
                                    "name": "weather_forecast",
                                    "arguments": {
                                        "location": "Манжерок",
                                        "num_days": 10,
                                    },
                                },
                                "functions_state_id": "state-2",
                            },
                            "finish_reason": "function_call",
                        }
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.make_client(handler, temp_dir) as client:
                result = client.step("v1-function-call", model="model", messages=[])

            self.assertEqual(result.text, "")
            self.assertEqual(result.function_calls[0]["role"], "assistant")
            self.assertEqual(
                result.function_calls[0]["function_call"]["arguments"]["num_days"],
                10,
            )
            self.assertEqual(result.function_calls[0]["functions_state_id"], "state-2")

    def test_history_messages_include_user_and_model_response(self):
        def handler(request):
            body = json.loads(request.content)
            if len(body["messages"]) == 1:
                return httpx.Response(
                    200,
                    json={
                        "messages": [
                            {
                                "role": "assistant",
                                "content": [{"text": "Первый ответ"}],
                            }
                        ]
                    },
                )
            self.assertEqual(body["messages"][0]["role"], "user")
            self.assertEqual(body["messages"][1]["role"], "assistant")
            return httpx.Response(
                200,
                json={
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [{"text": "Второй ответ"}],
                        }
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.make_client(handler, temp_dir) as client:
                first = client.step(
                    "first-history",
                    model="model",
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": "Первый вопрос"}],
                        }
                    ],
                )
                second = client.step(
                    "second-history",
                    model="model",
                    messages=first.history_messages
                    + [
                        {
                            "role": "user",
                            "content": [{"text": "Уточнение"}],
                        }
                    ],
                )

            self.assertEqual(first.request_texts_by_role["user"], "Первый вопрос")
            self.assertEqual(first.response_texts_by_role["assistant"], "Первый ответ")
            self.assertEqual(first.history_texts_by_role["user"], "Первый вопрос")
            self.assertEqual(first.history_texts_by_role["assistant"], "Первый ответ")
            self.assertEqual(
                [message["role"] for message in first.history_messages],
                ["user", "assistant"],
            )
            self.assertTrue((first.run_dir / "response_messages.json").exists())
            self.assertTrue((first.run_dir / "history_messages.json").exists())
            self.assertEqual(second.text, "Второй ответ")

    def test_history_messages_exclude_reasoning_by_default(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "messages": [
                        {
                            "role": "reasoning",
                            "content": [{"text": "Скрытое рассуждение"}],
                        },
                        {
                            "role": "assistant",
                            "content": [{"text": "Итог"}],
                        },
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.make_client(handler, temp_dir) as client:
                result = client.step(
                    "reasoning-history",
                    model="model",
                    messages=[
                        {
                            "role": "user",
                            "content": [{"text": "Вопрос"}],
                        }
                    ],
                )

            self.assertEqual(
                [message["role"] for message in result.response_messages],
                ["reasoning", "assistant"],
            )
            self.assertEqual(
                [message["role"] for message in result.history_messages],
                ["user", "assistant"],
            )
            self.assertEqual(
                [message["role"] for message in result.full_history_messages],
                ["user", "reasoning", "assistant"],
            )
            self.assertNotIn("reasoning", result.history_texts_by_role)
            self.assertEqual(result.reasoning_text, "Скрытое рассуждение")

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

    def test_plan_saves_loads_and_runs_selected_sequence(self):
        seen = []

        def handler(request):
            body = json.loads(request.content)
            seen.append(body["marker"])
            return httpx.Response(
                200,
                json={
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [{"text": body["marker"]}],
                        }
                    ]
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            plan_path = Path(temp_dir) / "plan.json"
            plan = LLMPlan()
            plan.add("first", model="model", messages=[], marker="one")
            plan.add(
                "second",
                payload={"model": "model", "messages": [], "marker": "two"},
            )
            plan.save(plan_path)

            loaded = LLMPlan.load(plan_path)
            with self.make_client(handler, temp_dir) as client:
                results = loaded.run(client, sequence=["second", "first"])

            self.assertEqual(loaded.names, ["first", "second"])
            self.assertEqual(seen, ["two", "one"])
            self.assertEqual(list(results), ["second", "first"])
            self.assertEqual(results["second"].text, "two")
            self.assertEqual(results["first"].text, "one")


if __name__ == "__main__":
    unittest.main()
