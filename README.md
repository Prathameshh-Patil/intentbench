# IntentBench

Three ways to classify short user messages into 150 intents (TF-IDF + logistic regression, fine-tuned DistilBERT, and an LLM), measured on the same data for accuracy, speed and cost.

> Work in progress. Results, live demo and setup instructions will appear here as the project is built.

## Dataset

[CLINC150](https://huggingface.co/datasets/clinc_oos) (`plus` configuration): short user queries labelled with 150 intents plus out-of-scope.

## Run locally

Requires Python 3.11+ (3.12 recommended).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m intentbench.data   # downloads CLINC150 and prints the split sizes
pytest -q                    # run the tests
```

## License

MIT
