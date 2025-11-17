# Financial Sentiment Dashboard

End-to-end workflow for cleansing Polygon news data, training two sentiment models, and serving predictions through an interactive Streamlit dashboard. The project satisfies the requirements outlined in **Task for AI Developer.pdf**:

- **Data layer** – cleans and maps Polygon news/insight feed into labelled text.
- **Model layer** – BERT + XGBoost baseline plus a fine-tuned FinBERT classifier.
- **Visualization layer** – dashboard with model switcher, single-text analysis, dataset analytics, sentiment distribution, sentiment vs time (percent view & timeline), and downloadable results.

---

## Repository structure

```
├── finbert_finetuned.py          # Fine-tune FinBERT with Optuna + early stopping
├── run_notebook_cells.py         # BERT embedding + XGBoost pipeline
├── inference.py                  # XGBoost inference helper
├── inference_finbert.py          # FinBERT inference helper (loads final_model/)
├── streamlit_app.py              # Dashboard entry point
├── finbert_models/
│   └── final_model/              # Tokenizer/config/weights/metadata (model.safetensors)
├── models/
│   ├── best_xgboost_model.json
│   └── model_metadata.json
├── raw_dataset.csv               # Prepared Polygon insight dataset
├── raw_dataset.py                # Script used to build raw_dataset.csv from JSON
├── README.md
├── pyproject.toml
└── Task for AI Developer.pdf
```

> **Note** The heavy training checkpoints were removed; only the finalized FinBERT weights/tokenizer remain under `finbert_models/final_model/`.

---

## Environment (Poetry)

```bash
# Install dependencies
poetry install

# Activate virtualenv
poetry shell

# (Optional) run commands directly
poetry run python <script.py>
```

Dependencies are declared in `pyproject.toml` (Python 3.10–3.13, PyTorch, Transformers, XGBoost, Optuna, Streamlit, etc.).

---

## Data preparation

1. **Build / refresh dataset**
   ```bash
   poetry run python raw_dataset.py
   # produces raw_dataset.csv
   ```

2. **EDA notebook (optional)**  
   `eda.ipynb` inspects class balance, ticker counts, etc.

---

## Model training

### 1. BERT embeddings + XGBoost
```bash
poetry run python run_notebook_cells.py
```
Steps performed:
- clean text → lowercase, strip URLs/special chars
- load `bert-base-uncased`, compute mean-pooled embeddings
- train/test split (stratified)
- baseline XGBoost, then Optuna search (multi:softprob)  
- save artifacts to `models/`:
  - `best_xgboost_model.json`
  - `model_metadata.json`
  - `text_cleaning.py`

### 2. FinBERT fine-tuning
```bash
poetry run python finbert_finetuned.py
```
- uses `yiyanghkust/finbert-tone`
- Optuna hyperparameter optimization (learning rate, batch size, max length, etc.)
- 50-epoch final training with early stopping (patience=5)
- saves tokenizer/config/metadata and `model.safetensors` to `finbert_models/final_model/`

---

## Inference utilities

| Script               | Description                                                         |
|----------------------|---------------------------------------------------------------------|
| `inference.py`       | Loads BERT tokenizer + XGBoost weights. CLI for quick predictions.  |
| `inference_finbert.py` | Loads tokenizer/config from `finbert_models/final_model/` and applies `model.safetensors` weights. Handles label mapping directly from the saved config. |

Example:
```bash
poetry run python - <<'PY'
from inference import SentimentPredictor
pred = SentimentPredictor()
print(pred.predict("Market surges on earnings", True))
PY
```

---

## Streamlit dashboard

```bash
poetry run streamlit run streamlit_app.py
```

Features:
- **Model picker** – XGBoost vs fine-tuned FinBERT.
- **Single-text analysis** – cleans input, displays sentiment, confidence, and probability bars.
- **Dataset analysis** – runs inference on full dataset or ticker subset:
  - sentiment distribution (pie + annotated bar)
  - sentiment over time (percentage lines)
  - sentiment timeline (stacked area)
  - **Sentiment vs Time** scatter with ticker & date range filters
  - downloadable CSV of scored rows

---

## Findings & Observations

- Dataset (13.3k rows) is heavily skewed toward positive news; class weights or stratification are critical.
- BERT + XGBoost baseline reached ~93% accuracy after Optuna tuning.
- FinBERT fine-tuning on the same corpus achieved ~98% test accuracy (metadata stored alongside the model).
- Qualitatively, FinBERT captures subtle finance phrasing (e.g., “guidance maintained” → neutral/positive) better than the baseline.

---

## Deployment checklist

1. Ensure only the trimmed artifacts remain (no local venvs, checkpoints, or notebooks with secrets).
2. Commit `pyproject.toml`, `.gitignore`, trained model folders, scripts, and README.
3. Push to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: financial sentiment dashboard"
   git remote add origin <YOUR_REPO_URL>
   git push -u origin main
   ```

---

## Troubleshooting

| Issue | Resolution |
|-------|------------|
| `torch._environment PermissionError` | Run commands outside sandbox (already handled in this repo). |
| FinBERT predictions off | Ensure `finbert_models/final_model/` contains `config.json`, `tokenizer.json`, `model.safetensors`, and `model_metadata.json`. |
| Optuna runs out of memory | Reduce `max_length` or `batch_size` in `finbert_finetuned.py` `objective()` hyperparameters. |

---

## License

Not specified – add your preferred license before publishing.

