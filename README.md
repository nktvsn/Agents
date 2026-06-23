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
print(result.usage)
print(result.execution_steps)
print(result.run_dir)
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

```python
def print_event(event, data):
    if event == "response.message.delta":
        for choice in data.get("choices", []):
            print(choice.get("delta", {}).get("content", ""), end="")

result = client.step(
    "stream-call",
    model="GigaChat-2-Max",
    messages=[{"role": "user", "content": [{"text": "Напиши три пункта."}]}],
    stream=True,
    on_event=print_event,
)
```

Все события сохраняются в `events.jsonl`.

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
- `text.txt`;
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

