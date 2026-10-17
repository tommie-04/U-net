# IMDb Review Classification

A Python project for binary sentiment classification of movie reviews.

The current implementation loads and inspects the IMDb training split. Model training, evaluation, and prediction will be added incrementally.

## Setup

Use Python 3.12. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Inspect the data

```bash
python scripts/inspect_data.py
```

The script prints the dataset summary, the number of training reviews, and the first sample. The returned training split contains 25,000 reviews with two fields:

- `text`: the review text.
- `label`: `0` for negative sentiment and `1` for positive sentiment.

The first run downloads and caches the dataset through Hugging Face Datasets. Dataset preparation may include other splits, while the script returns and inspects only `train`.

## Data

The project uses the [Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), loaded as `stanfordnlp/imdb`. Dataset files and the local virtual environment are excluded from version control.
