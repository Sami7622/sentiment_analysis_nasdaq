# Financial Sentiment Dashboard

End-to-end workflow for Sentiment analysis on financial data, training two sentiment models, and serving predictions through an interactive Streamlit dashboard. The project satisfies the outlined requirements:

- **Data layer** – clean, perform EDA and maps into labelled data.
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
## Prerequisites
- Python 3.10-3.13
- Poetry (Install via 'pipx install poetry' or 'curl -sSL https://install.python-poetry.org | python3 -'

## FinBERT model weights

`finbert_models/final_model/model.safetensors` is ~420 MB and not stored in Git.  
Before running FinBERT inference or the Streamlit dashboard’s FinBERT option, download the weights and place them in that folder:

1. Download from: repo's finbert_models/final_model/model.safetensors.
2. Place the file at `finbert_models/final_model/model.safetensors`.
3. Verify it’s ~420 MB: `ls -lh finbert_models/final_model/model.safetensors`.

> Tip: if you clone via Git LFS, run `git lfs install && git lfs pull` to fetch the weights automatically.


## Environment (Poetry)

```bash
# Install dependencies
# Install Poetry once (if not already)
pipx install poetry                     # or: curl -sSL https://install.python-poetry.org | python3 -

# Optional: enable `poetry shell`
poetry self add poetry-plugin-shell

poetry install                          # resolves & installs deps + creates venv

# Install shell plugin
poetry self add poetry-plugin-shell

# Activate virtualenv
poetry shell


# Run the streamlit Application 
poetry run streamlit run streamlit_app.py



# (Optional) not needed currently - run commands directly
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

## FinBERT weights

The fine-tuned weights are large (~420 MB) and not committed to Git.

1. Obtain `model.safetensors` from the training machine (or your preferred storage).  
2. Place it at `finbert_models/final_model/model.safetensors`.  
3. Verify the size: `ls -lh finbert_models/final_model/model.safetensors`.

> Tip: if you prefer to keep weights in the repo, add Git LFS tracking (`git lfs track "finbert_models/final_model/*.safetensors"`), then push via LFS.

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
  - Use exmaple tickers like 'AMZN' 'AAPL' to test.

---

## Findings & Observations

- BERT + XGBoost baseline reached ~93% accuracy after Optuna tuning.
- FinBERT fine-tuning on the same corpus achieved ~98% test accuracy (metadata stored alongside the model).
- Qualitatively, FinBERT captures subtle finance phrasing (e.g., “guidance maintained” → neutral/positive) better than the baseline and might need more data for better differencing.

