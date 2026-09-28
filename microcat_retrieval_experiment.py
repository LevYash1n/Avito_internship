"""Проверить, добавляет ли поиск внутри предсказанных microcat релевантные ID.

Запуск: python microcat_retrieval_experiment.py
Функции очистки и BM25 берутся из experiments.ipynb, чтобы сравнение
совпадало с текущим baseline. Этот скрипт не читает benchmark-разметку.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from microcat_experiment import DATA_DIR, QUERY_COLS, make_splits, query_text


def top_rows(scores, rows, k):
    """Индексы top-k среди заданных строк корпуса."""
    if len(rows) <= k:
        selected = rows
    else:
        selected = rows[np.argpartition(scores[rows], -k)[-k:]]
    return selected[np.lexsort((selected, -scores[selected]))]


def main():
    notebook = json.loads((DATA_DIR / "experiments.ipynb").read_text(encoding="utf-8"))

    def notebook_source(cell):
        return "".join(notebook["cells"][cell]["source"])

    pairs = pd.read_parquet(
        DATA_DIR / "train.parquet",
        columns=QUERY_COLS + ["item_id", "item_microcat_id"],
    ).drop_duplicates(QUERY_COLS + ["item_id"])
    signatures, pairs = make_splits(pairs)
    benchmark_corpus = pd.read_parquet(DATA_DIR / "benchmark_items.parquet")
    item_row_by_id = pd.Series(
        np.arange(len(benchmark_corpus), dtype=np.int32),
        index=benchmark_corpus["item_id"],
    )
    assert item_row_by_id.index.is_unique
    benchmark_pairs = pairs.loc[pairs["item_id"].isin(item_row_by_id.index)]
    relevant_rows = benchmark_pairs.assign(
        item_row=benchmark_pairs["item_id"].map(item_row_by_id)
    ).groupby("query_row")["item_row"].agg(set)
    queries = signatures.loc[
        signatures["query_row"].isin(relevant_rows.index)
        & signatures["signature_split"].isin(("val", "test"))
    ].copy()
    queries["validation_query_id"] = queries["query_row"].astype(str)
    queries = queries.reset_index(drop=True)
    relevant_sets = [relevant_rows.at[row] for row in queries["query_row"]]
    print(f"Eligible queries: {len(queries):,}", flush=True)

    # Обучаем классификатор microcat только на train-сигнатурах.
    fit_pairs = pairs.loc[pairs["signature_split"].eq("train")]
    microcat_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=3, max_features=200_000,
        sublinear_tf=True, dtype=np.float32,
    )
    microcat_fit = microcat_vectorizer.fit_transform(query_text(fit_pairs))
    microcat_model = MultinomialNB(alpha=0.1).fit(
        microcat_fit, fit_pairs["item_microcat_id"],
    )
    del microcat_fit
    fit_item_ids = set(fit_pairs["item_id"])
    cold_pairs = benchmark_pairs.loc[~benchmark_pairs["item_id"].isin(fit_item_ids)]
    cold_relevant_by_query = cold_pairs.assign(
        item_row=cold_pairs["item_id"].map(item_row_by_id)
    ).groupby("query_row")["item_row"].agg(set)
    cold_relevant_sets = [
        cold_relevant_by_query.get(row, set()) for row in queries["query_row"]
    ]
    known_train_texts = set(query_text(fit_pairs))
    known_query_text = query_text(queries).isin(known_train_texts).to_numpy()
    microcat_proba = microcat_model.predict_proba(
        microcat_vectorizer.transform(query_text(queries))
    )
    microcat_order = microcat_model.classes_[
        np.argsort(-microcat_proba, axis=1)[:, :5]
    ]
    corpus_microcats = benchmark_corpus["item_microcat_id"].to_numpy()
    microcat_rows = {
        cat: np.flatnonzero(corpus_microcats == cat).astype(np.int32)
        for cat in np.unique(corpus_microcats)
    }

    # Те же функции подготовки текста, BM25 и exact-title, что в baseline.
    def quiet_tqdm(iterable, **kwargs):
        print("  " + kwargs.get("desc", "work"), flush=True)
        return iterable

    tqdm.tqdm = quiet_tqdm
    globals()["benchmark_corpus"] = benchmark_corpus
    code = notebook_source(13)
    assert "vectorizer = TfidfVectorizer(" in code
    exec(code.split("vectorizer = TfidfVectorizer(")[0], globals())
    exec(notebook_source(19), globals())
    bm25_items = make_bm25_matrix(item_count_matrix)
    globals().pop("item_count_matrix")
    bm25_queries = bm25_vectorizer.transform(prepare_query_text(queries)).tocsr()
    title_vectorizer = TfidfVectorizer(
        ngram_range=(1, 1), min_df=2, max_features=200_000,
        dtype=np.float32,
    )
    title_items = title_vectorizer.fit_transform(
        normalize_for_title_match(benchmark_corpus["item_title_raw"])
    ).tocsr()
    title_queries = title_vectorizer.transform(
        normalize_for_title_match(queries["search_query"])
    ).tocsr()
    exact = get_exact_title_predictions(
        queries, benchmark_corpus, bm25_queries, bm25_items, top_k=200,
    )
    exact_rows = [
        set(item_row_by_id.loc[exact[query_id]])
        for query_id in queries["validation_query_id"]
    ]

    item_rows = np.arange(len(benchmark_corpus), dtype=np.int32)
    names = (
        "BM25 top50", "microcat top1 BM25 top50", "microcat 30/15/5 BM25",
        "BM25+exact pool", "BM25+exact+title pool",
        "+microcat top1 x200 pool", "+microcat top3 x100 pool",
        "+microcat top3 x200 pool", "+microcat top5 x200 pool",
        "+microcat top3 x500 pool", "+microcat top3 x1000 pool",
        "+microcat top5 x1000 pool",
        "+microcat top3 x200 title pool",
        "+microcat top3 x200 BM25+title pool",
    )
    totals = {split: dict.fromkeys(names, 0.0) for split in ("val", "test")}
    counts = {split: 0 for split in totals}
    bucket_names = ("BM25+exact+title pool", "+microcat top3 x200 pool", "+microcat top3 x1000 pool")
    bucket_totals = {
        (split, seen): dict.fromkeys(bucket_names, 0.0)
        for split in ("val", "test") for seen in (False, True)
    }
    bucket_counts = {key: 0 for key in bucket_totals}
    cold_totals = {
        (split, fresh_text): dict.fromkeys(bucket_names, 0.0)
        for split in ("val", "test") for fresh_text in (False, True)
    }
    cold_counts = {key: 0 for key in cold_totals}
    query_splits = queries["signature_split"].to_numpy()
    for start in range(0, len(queries), 32):
        stop = min(start + 32, len(queries))
        bm25_scores = (bm25_queries[start:stop] @ bm25_items.T).toarray()
        title_scores = (title_queries[start:stop] @ title_items.T).toarray()
        for pos in range(start, stop):
            local = pos - start
            scores = bm25_scores[local]
            global_200 = top_rows(scores, item_rows, 200)
            title_200 = top_rows(title_scores[local], item_rows, 200)
            cat_top200 = [
                top_rows(scores, microcat_rows.get(cat, np.empty(0, dtype=np.int32)), 1000)
                for cat in microcat_order[pos]
            ]
            cat_title_top200 = [
                top_rows(title_scores[local], microcat_rows.get(cat, np.empty(0, dtype=np.int32)), 200)
                for cat in microcat_order[pos][:3]
            ]
            base = set(global_200) | exact_rows[pos]
            with_title = base | set(title_200)
            # Простой итоговый top-50 до обучения нового reranker.
            micro1 = list(cat_top200[0][:50])
            mixed = list(dict.fromkeys([
                *cat_top200[0][:30], *cat_top200[1][:15],
                *cat_top200[2][:5], *global_200,
            ]))
            candidate_sets = {
                "BM25 top50": set(global_200[:50]),
                "microcat top1 BM25 top50": set((micro1 + list(global_200))[:50]),
                "microcat 30/15/5 BM25": set(mixed[:50]),
                "BM25+exact pool": base,
                "BM25+exact+title pool": with_title,
                "+microcat top1 x200 pool": with_title | set(cat_top200[0][:200]),
                "+microcat top3 x100 pool": with_title | set().union(*(set(x[:100]) for x in cat_top200[:3])),
                "+microcat top3 x200 pool": with_title | set().union(*(set(x[:200]) for x in cat_top200[:3])),
                "+microcat top5 x200 pool": with_title | set().union(*(set(x[:200]) for x in cat_top200[:5])),
                "+microcat top3 x500 pool": with_title | set().union(*(set(x[:500]) for x in cat_top200[:3])),
                "+microcat top3 x1000 pool": with_title | set().union(*(set(x) for x in cat_top200[:3])),
                "+microcat top5 x1000 pool": with_title | set().union(*(set(x) for x in cat_top200[:5])),
                "+microcat top3 x200 title pool": with_title | set().union(*(set(x) for x in cat_title_top200)),
                "+microcat top3 x200 BM25+title pool": (
                    with_title | set().union(*(set(x[:200]) for x in cat_top200[:3]))
                    | set().union(*(set(x) for x in cat_title_top200))
                ),
            }
            split = query_splits[pos]
            if split in totals:
                counts[split] += 1
                bucket_key = (split, bool(known_query_text[pos]))
                bucket_counts[bucket_key] += 1
                relevant = relevant_sets[pos]
                for name, candidates in candidate_sets.items():
                    value = len(candidates & relevant) / len(relevant)
                    totals[split][name] += value
                    if name in bucket_totals[bucket_key]:
                        bucket_totals[bucket_key][name] += value
                cold_relevant = cold_relevant_sets[pos]
                if cold_relevant:
                    cold_key = (split, not bool(known_query_text[pos]))
                    cold_counts[cold_key] += 1
                    for name in bucket_names:
                        cold_totals[cold_key][name] += (
                            len(candidate_sets[name] & cold_relevant) / len(cold_relevant)
                        )
        if stop % 1024 < 32 or stop == len(queries):
            print(f"Scored {stop:,}/{len(queries):,} queries", flush=True)

    for split in ("val", "test"):
        print(f"{split}: {counts[split]:,} queries", flush=True)
        for name in names:
            value = totals[split][name] / counts[split]
            print(f"  {name}: {value:.5f}", flush=True)
        for seen in (True, False):
            key = (split, seen)
            parts = ", ".join(
                f"{name}={bucket_totals[key][name] / bucket_counts[key]:.5f}"
                for name in bucket_names
            )
            print(f"  {'known' if seen else 'new'} text ({bucket_counts[key]:,}): {parts}", flush=True)
        for fresh_text in (False, True):
            key = (split, fresh_text)
            parts = ", ".join(
                f"{name}={cold_totals[key][name] / cold_counts[key]:.5f}"
                for name in bucket_names
            )
            label = "new text + new item" if fresh_text else "new item"
            print(f"  {label} ({cold_counts[key]:,}): {parts}", flush=True)


if __name__ == "__main__":
    main()
