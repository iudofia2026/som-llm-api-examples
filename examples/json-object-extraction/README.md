# JSON object extraction

Shows practical JSON extraction patterns:

- `response_format={"type": "json_object"}` for long/open-ended extraction
- two-pass notes -> JSON conversion
- strict bounded `json_schema` for small schemas

Offline validation:

```sh
python3 example.py --self-test
```

Live run:

```sh
SOM_LLM_KEY=sk-som-... ./example.py --live
```
