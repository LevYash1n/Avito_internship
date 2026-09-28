"""Проверка идеи: предсказывать подкатегорию услуги из текста запроса.

Скрипт использует тот же signature split, что и experiments.ipynb. Обучение
видит только train-сигнатуры; val/holdout разметка нужна лишь для оценки.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


DATA_DIR = Path(__file__).resolve().parent
QUERY_COLS = [
    "search_query", "search_location_id", "search_is_delivery_search",
    "search_infm_params_text", "search_category",
]
SPLIT_SEED = 43


def query_text(frame):
    """Текст и фильтры — только признаки запроса, без сведений об ответе."""
    text = frame["search_query"].fillna("").astype(str)
    filters = frame["search_infm_params_text"].fillna("").astype(str)
    return (
        (text + " " + filters)
        .str.lower()
        .str.replace("ё", "е", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def make_splits(pairs):
    # В точности повторяем порядок сигнатур и random_state из ноутбука.
    signatures = pairs[QUERY_COLS].drop_duplicates().reset_index(drop=True)
    order = signatures.sample(frac=1, random_state=SPLIT_SEED).index
    n_train = int(len(signatures) * 0.70)
    n_val = int(len(signatures) * 0.15)
    signatures["signature_split"] = "test"
    signatures.loc[order[:n_train], "signature_split"] = "train"
    signatures.loc[order[n_train:n_train + n_val], "signature_split"] = "val"
    signatures.insert(0, "query_row", np.arange(len(signatures), dtype=np.int32))
    return signatures, pairs.merge(
        signatures[QUERY_COLS + ["query_row", "signature_split"]],
        on=QUERY_COLS, how="left", validate="many_to_one",
    )


def main():
    pairs = pd.read_parquet(
        DATA_DIR / "train.parquet",
        columns=QUERY_COLS + ["item_id", "item_microcat_id"],
    ).drop_duplicates(QUERY_COLS + ["item_id"])
    signatures, pairs = make_splits(pairs)
    items = pd.read_parquet(
        DATA_DIR / "benchmark_items.parquet",
        columns=["item_id", "item_microcat_id"],
    )
    assert items["item_id"].is_unique

    # В eval учитываются только известные позитивы из настоящего корпуса.
    eval_pairs = pairs.merge(
        items, on="item_id", how="inner", suffixes=("_train", "_corpus"),
        validate="many_to_one",
    )
    assert (
        eval_pairs["item_microcat_id_train"]
        == eval_pairs["item_microcat_id_corpus"]
    ).all()

    fit_pairs = pairs.loc[pairs["signature_split"].eq("train")]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_features=200_000,
        sublinear_tf=True, dtype=np.float32,
    )
    print(f"Train pairs: {len(fit_pairs):,}; classes: {fit_pairs['item_microcat_id'].nunique()}", flush=True)
    fit_matrix = vectorizer.fit_transform(query_text(fit_pairs))
    model = MultinomialNB(alpha=0.1)
    model.fit(fit_matrix, fit_pairs["item_microcat_id"])
    del fit_matrix

    known_train_texts = set(query_text(fit_pairs))
    for split in ("val", "test"):
        relevant = eval_pairs.loc[eval_pairs["signature_split"].eq(split)]
        query_rows = relevant["query_row"].drop_duplicates().to_numpy()
        queries = signatures.set_index("query_row").loc[query_rows].reset_index()
        texts = query_text(queries)
        probability = model.predict_proba(vectorizer.transform(texts))
        rank = np.argsort(-probability, axis=1)
        classes = model.classes_[rank]
        query_pos = pd.Series(np.arange(len(query_rows)), index=query_rows)
        relevant_positions = relevant["query_row"].map(query_pos).to_numpy()
        relevant_classes = relevant["item_microcat_id_corpus"].to_numpy()
        n_rel = relevant.groupby("query_row").size().reindex(query_rows).to_numpy()
        seen_text = texts.isin(known_train_texts).to_numpy()
        print(
            f"{split}: {len(query_rows):,} queries, {len(relevant):,} positive pairs; "
            f"known text {seen_text.mean():.3f}", flush=True,
        )
        for k in (1, 3, 5, 10):
            hit = np.any(
                classes[relevant_positions, :k] == relevant_classes[:, None],
                axis=1,
            )
            numerator = np.bincount(relevant_positions, weights=hit, minlength=len(query_rows))
            recall_per_query = numerator / n_rel
            print(
                f"  microcat top-{k}: all {recall_per_query.mean():.5f}; "
                f"known {recall_per_query[seen_text].mean():.5f}; "
                f"new {recall_per_query[~seen_text].mean():.5f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
