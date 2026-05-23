# Python examples

Set your key once:

```sh
export SOM_LLM_KEY=sk-som-...
```

Optional: pin a model. If omitted, the examples ask `/v1/models` and use the first advertised model.

```sh
export SOM_LLM_MODEL=Qwen3.5-122B-A10B-FP8
```

Run any example directly:

```sh
./01_chat.py
./02_stream.py
./03_classify.py
./04_tag.py
./05_extract_json.py
./06_thinking.py
```

## What each example shows

- `01_chat.py` — normal OpenAI-style chat completion.
- `02_stream.py` — streaming tokens as they arrive.
- `03_classify.py` — single-label classification with a regex constraint.
- `04_tag.py` — multi-label tagging with `json_object`.
- `05_extract_json.py` — typed JSON extraction validated by Pydantic.
- `06_thinking.py` — enabling Qwen thinking for multi-step reasoning.

For short classification/extraction jobs, the examples disable thinking:

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

For harder reasoning, turn thinking on and give the model enough `max_tokens` for reasoning plus the final answer.
