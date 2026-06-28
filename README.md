# Notebook LLM Client

Небольшой синхронный клиент для экспериментов с LLM API из Jupyter:

- mTLS через пути к PEM-сертификату и ключу;
- проверка сертификата сервера отключена;
- заголовок `Authorization` не создается;
- любые поля API можно передать без клиентской валидации;
- запросы, ответы, метаданные, промежуточные шаги и SSE-события сохраняются на диск.

## Настройка

```bash
export LLM_BASE_URL="https://example.internal"
export LLM_CERT="/path/to/client.crt"
export LLM_KEY="/path/to/client.key"
```

Установка в kernel:

```python
%pip install -e .
```

## Быстрый вызов

```python
from llm_client import LLMClient

client = LLMClient.from_env(runs_dir="runs")

result = client.step(
    "first-call",
    model="GigaChat-2-Max",
    messages=[
        {
            "role": "user",
            "content": [{"text": "Кратко сравни REST и gRPC."}],
        }
    ],
    model_options={
        "temperature": 0.2,
        "max_tokens": 500,
    },
)

print(result.text)
print(result.assistant_text)
print(result.reasoning_text)
print(result.function_calls)
print(result.usage)
print(result.execution_steps)
print(result.run_dir)
```

`result.text` является коротким алиасом для `result.assistant_text`. Сообщения
с ролью `reasoning` не смешиваются с финальным ответом и доступны через
`result.reasoning_text`. Вызовы функций доступны через
`result.function_calls` и `result.function_call_text`.

Для сообщений есть единый метод:

```python
result.messages()              # conversation: история без system и reasoning
result.messages("request")     # сообщения, которые ушли в запрос
result.messages("response")    # сообщения, которые вернула модель
result.messages("history")     # request + response без reasoning
result.messages("full_history")# request + response полностью

result.role_texts()            # тексты по ролям для conversation
result.role_texts("response")  # тексты по ролям для ответа модели
```

Поля `model`, `messages`, `model_options`, `tools` и любые будущие параметры
передаются в `step()` или `chat()` как есть.

Готовый словарь запроса тоже можно передать целиком:

```python
result = client.chat(name="raw-payload", payload=my_payload)
```

При совпадении ключей аргументы после `payload` имеют приоритет:

```python
result = client.chat(
    name="override",
    payload=my_payload,
    model="another-model",
)
```

## Потоковый ответ

В Jupyter callback вызывается при получении каждого SSE-события. Готовый
printer понимает V2-сообщения с ролями `reasoning` и `assistant`:

```python
from llm_client import JupyterStreamPrinter

printer = JupyterStreamPrinter(show_reasoning=True)
result = client.step(
    "stream-call",
    model="GigaChat-2-Max",
    messages=[{"role": "user", "content": [{"text": "Напиши три пункта."}]}],
    model_options={"reasoning": {"effort": "medium"}},
    stream=True,
    on_event=printer,
)

print("\n\nassistant:", result.assistant_text)
print("reasoning:", result.reasoning_text)
```

Вызов блокирует текущую ячейку до завершения ответа, но текст появляется
постепенно. Прервать запрос можно кнопкой остановки kernel. Все сырые события
сохраняются в `events.jsonl`.

## Вызовы функций

Если модель вернула аргументы для вызова функции, клиент сохраняет их отдельно:

```python
result = client.step(
    "weather-call",
    model="GigaChat-2-Pro",
    messages=[{"role": "user", "content": [{"text": "Погода в Манжероке"}]}],
    functions=[
        {
            "name": "weather_forecast",
            "description": "Возвращает прогноз погоды",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "num_days": {"type": "integer"},
                },
                "required": ["location"],
            },
        }
    ],
    function_call="auto",
)

print(result.function_calls)
print(result.function_call_text)
```

Для V2-сообщений `function_call` извлекается из `content[].function_call`.
Для совместимого формата он извлекается из `choices[].message.function_call`.
На диск дополнительно пишутся `function_calls.json` и `function_call.txt`.

## История сообщений

После любого шага можно взять готовую историю и передать ее в следующий вызов.
По умолчанию `messages()` возвращает conversation-историю: там уже нет
`reasoning` и `system`.

```python
first = client.step(
    "first",
    model="GigaChat-2-Max",
    messages=[
        {
            "role": "user",
            "content": [{"text": "Сформулируй короткую гипотезу"}],
        }
    ],
)

second = client.step(
    "second",
    model="GigaChat-2-Max",
    messages=first.messages()
    + [
        {
            "role": "user",
            "content": [{"text": "Теперь проверь ее на слабые места"}],
        }
    ],
)
```

`messages("history")` не включает сообщения с ролью `reasoning`, чтобы не
прокидывать внутренние рассуждения модели в следующий запрос. `messages()` или
`messages("conversation")` дополнительно исключает `system`, чтобы системный
промпт не дублировался при следующем вызове.

Если нужен явный контроль ролей:

```python
messages = first.messages("full_history", exclude_roles={"system", "reasoning"})
texts = first.role_texts("full_history", exclude_roles={"system"})
```

Если нужен полный след для анализа, используйте `messages("full_history")` или
`messages("response")`.

Тексты по ролям можно использовать для условий и переменных:

```python
user_text = first.role_texts()["user"]
assistant_text = first.role_texts()["assistant"]
function_call = first.role_texts().get("function_call", "")
```

История дополнительно сохраняется в `history_messages.json` и
`conversation_messages.json`.

## Предсохраненные шаги

Можно заранее собрать набор шагов, сохранить их в JSON, а потом запускать в
любой нужной последовательности:

```python
from llm_client import LLMPlan

plan = LLMPlan()
plan.add(
    "extract-facts",
    model="GigaChat-2-Max",
    messages=[{"role": "user", "content": [{"text": "Вытащи факты из текста"}]}],
    model_options={"temperature": 0},
)
plan.add(
    "write-summary",
    model="GigaChat-2-Max",
    messages=[{"role": "user", "content": [{"text": "Сделай краткое резюме"}]}],
    model_options={"temperature": 0.2},
)

plan.save("plans/demo.json")
```

В другой ячейке или в другой день:

```python
plan = LLMPlan.load("plans/demo.json")

results = plan.run(client, sequence=["write-summary", "extract-facts"])

print(results["write-summary"].assistant_text)
print(results["extract-facts"].run_dir)
```

Один шаг можно выполнить отдельно и временно переопределить любые параметры:

```python
result = plan.run_one(
    client,
    "write-summary",
    model_options={"temperature": 0.7},
)
```

## Остальные endpoints

```python
models = client.models()

vectors = client.embeddings(
    model="Embeddings",
    input=["Первый текст", "Второй текст"],
)

custom = client.request(
    "POST",
    "/some/future/endpoint",
    name="custom",
    json={"any": {"payload": True}},
    params={"debug": "1"},
)
```

## Артефакты запуска

Каждый вызов создает `runs/<UTC timestamp>-<name>/`:

- `request.json`;
- `response.json`;
- `metadata.json`;
- `response_messages.json`;
- `history_messages.json`;
- `conversation_messages.json`;
- `text.txt`;
- `assistant.txt`;
- `reasoning.txt`, если модель вернула reasoning;
- `function_calls.json` и `function_call.txt`, если модель вернула вызов функции;
- `execution_steps.json`, если API их вернул;
- `events.jsonl` для streaming;
- `error.json` при ошибке.

Клиент нужно закрыть после работы:

```python
client.close()
```

Либо использовать context manager:

```python
with LLMClient.from_env() as client:
    result = client.step(...)
```

Отключенная проверка сертификата сервера снижает защищенность соединения и
должна использоваться только в доверенной сети.
