# Interview questions about IntentBench

The 15 questions most likely to come up, with honest answers based on what this repo actually contains. Numbers are from `reports/` and can be regenerated with the scripts listed in the README.

### 1. What does the project do, in one minute?

It compares three ways to classify short user messages into 150 intents plus out-of-scope, on the CLINC150 dataset: a TF-IDF + logistic regression baseline, a fine-tuned DistilBERT, and an LLM (Gemini, zero- and few-shot). All three use the same splits and the same metric code, and are measured on accuracy, macro-F1, out-of-scope detection, latency and cost. The best one, DistilBERT with a confidence threshold for out-of-scope, is served by a FastAPI app in Docker, and a Gradio demo shows all three side by side.

### 2. What was the result?

On the full 5,500-query test set, the served DistilBERT reached accuracy 0.918 and macro-F1 0.939. TF-IDF reached 0.843 and 0.877. On the 400-query sample all three were run on, accuracy was 0.910 for DistilBERT, 0.897 for the few-shot LLM and 0.838 for TF-IDF. DistilBERT takes 12 ms per request on CPU, against a 1.6 s median for the LLM, and costs nothing per request; the LLM costs $0.67 per 1,000.

### 3. Why did you keep the test set for the very end?

Every decision made by looking at a score fits the system a little to that data. If I tuned on test, the test score would stop predicting how the model does on new data. So all tuning (n-grams, C, the best epoch, the LLM variant, the out-of-scope threshold) used validation, and `scripts/compare.py` touched test once. The LLM answers are cached, so re-running it reproduces the same numbers without re-tuning.

### 4. Why macro-F1 and not just accuracy?

Accuracy is dominated by big classes. Macro-F1 computes F1 for each of the 150 intents and averages them equally, so a badly handled intent pulls it down. Out-of-scope is handled separately: it's 18% of test, so lumping it in would hide whether the model detects it. I report its precision and recall on their own.

### 5. Why was out-of-scope the hardest part, and what did you do about it?

Train is only 1.6% out-of-scope, while test is 18%. A classifier trained to pick one of 151 classes tends to choose the nearest intent ("add mary to my phone plan" becomes `sync_device`). I added a confidence threshold: if the top probability is below 0.69, the answer is out-of-scope. I chose it on validation as the highest out-of-scope recall costing at most 1 point of in-scope accuracy. On test it lifted out-of-scope recall from 0.59 to 0.78.

### 6. Why that threshold rule and not, say, maximizing accuracy?

Validation is only 3% out-of-scope, so accuracy barely changes with the threshold. I first tried maximizing the average of in-scope accuracy and out-of-scope recall. That picked 0.98 and rejected 6% of valid queries, because the model is over-confident. Capping the in-scope loss at 1 point is a product-style rule: "wrongly reject at most about 1 in 100 real requests". You can explain and change it.

### 7. What does fine-tuning actually change in DistilBERT?

We replace the pre-training head with a new, randomly initialized 151-way linear layer. Then we train with a small learning rate (5e-5), which trains the new head and slightly adjusts all 66M pre-trained weights toward this task. The pre-trained part already places "freeze my card" near "lock my card", which TF-IDF can't do, because it only matches surface text.

### 8. How did you spot overfitting?

Training loss fell to 0.005 while validation loss bottomed at epoch 4 (0.238) and rose to 0.254 by epoch 8. Validation macro-F1 still improved slightly. So the predictions stayed right, but the model became more confident on its mistakes. That's why the probabilities are over-confident and the threshold had to be tuned rather than set to 0.5.

### 9. How did you make the LLM output usable?

The prompt lists all 150 intent names, and `oos` is renamed `out_of_scope`, with an instruction not to force a match. The answer must be JSON `{"intent": ...}` through a response schema, and only an exact intent name is accepted. Anything else gets one retry with a correction, then counts as out-of-scope. Gemini rejects a 151-value enum, so it's strict parsing there; with Groq, the schema enforces the enum. On test there were 0 invalid answers.

### 10. How did you choose the few-shot examples?

Not by guessing from intent names. I ran the baseline on validation rows *outside* the LLM's scored sample, took its 15 most-confused pairs, and added 3 training examples for each intent involved, plus 10 out-of-scope examples. Few-shot beat zero-shot on validation (macro-F1 0.934 vs 0.916), so only few-shot ran on test.

### 11. How is the LLM evaluation reproducible if the model is non-deterministic?

Every response is cached on disk, keyed by a hash of model + system prompt + query. A re-run reads the cache: same predictions, zero cost. Changing the prompt changes the key, so stale answers can't leak in. It also made the free tier workable: when the daily quota (500 requests) ran out, the run resumed the next day from where it stopped.

### 12. Why was the LLM evaluated on only 400 test queries?

Cost and quota. The free tier allows 500 requests per day, and I chose not to pay. The brief allowed a fixed random sample, so all approaches are also reported on the same 400 rows. The downside is noise: macro-F1 on 400 rows rests on about 2 examples per intent, so on the sample I compare by accuracy, and the out-of-scope figures rest on 78 queries.

### 13. What were the main failure patterns?

First, out-of-scope queries close to a real domain get pulled into it (phone plans, buses, cars). The LLM, told not to force a match, missed far fewer of these (13 of 78 versus 36-37) but raised more false alarms (7 versus 1). Second, sibling intents that differ only in the action: `insurance` vs `insurance_change`, `redeem_rewards` vs `rewards_balance`. There is also some label noise: name-change requests labelled `pin_change`.

### 14. How does serving work, and how did you test it?

FastAPI loads the model once at startup (the lifespan hook) and exposes `POST /classify`, `GET /health` and `GET /model-info`. Pydantic validates input: 1-500 characters, not blank, otherwise 422. The response has the label, the confidence and the top 3 intents. Tests use a fake predictor injected through the same factory, so they check the HTTP contract and the threshold logic in milliseconds without the 257 MB model. The Docker image is CPU-only (about 570 MB) and runs as a non-root user.

### 15. What would you do next, or differently?

Calibrate the probabilities (temperature scaling) before choosing the threshold. Export to ONNX for a smaller, faster CPU image. Try retrieval-based few-shot examples for the LLM. Evaluate on a second dataset (BANKING77) to see whether the conclusions hold. Deploy the demo publicly: Hugging Face now charges for Gradio Spaces, so that needs a different free host or a paid plan.
