# Efficient client

Shows bounded concurrency, a small retry loop for transient failures, and reusing one HTTP client across requests.

Offline validation:

```sh
python3 example.py --self-test
```

Live run:

```sh
SOM_LLM_KEY=sk-som-... ./example.py --live
```
