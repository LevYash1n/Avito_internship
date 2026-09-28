"""Воспроизводимый запуск title_rank опыта из experiments.ipynb.

Скрипт не создаёт объединённый корпус и не считает неиспользуемые старые
baseline-модели, поэтому требует меньше памяти, чем полный запуск ноутбука.
"""

import json
import argparse
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from IPython.display import display


def main(with_dense=False):
    started = time.time()
    notebook = json.loads(Path("experiments.ipynb").read_text(encoding="utf-8"))

    def source(cell_id):
        cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
        return "".join(cell["source"])

    def run(cell_id, stop_before=None):
        code = source(cell_id)
        if stop_before:
            assert stop_before in code
            code = code.split(stop_before)[0]
        print(f"Cell {cell_id}", flush=True)
        exec(compile(code, f"experiments-cell-{cell_id}", "exec"), globals())

    def quiet_tqdm(iterable, **kwargs):
        print("  " + kwargs.get("desc", "work"), flush=True)
        return iterable

    original_tqdm = tqdm.tqdm
    tqdm.tqdm = quiet_tqdm
    globals().update({
        "DATA_DIR": Path.cwd(),
        "QUERY_COLS": [
            "search_query", "search_location_id", "search_is_delivery_search",
            "search_infm_params_text", "search_category",
        ],
        "SPLIT_SEED": 42,
        "SPLITS": ("train", "val", "test"),
        "validation_source": pd.read_parquet(
            "train.parquet", columns=[
                "search_query", "search_location_id", "search_is_delivery_search",
                "search_infm_params_text", "search_category", "item_id",
            ],
        ),
        "benchmark_corpus": pd.read_parquet("benchmark_items.parquet"),
    })
    globals()["benchmark_item_ids"] = frozenset(benchmark_corpus["item_id"])
    run("validation-build")
    run("validation-builder")
    globals()["benchmark_validation"] = build_validation_dataframe(benchmark_corpus)
    for split, suffix in (("train", "train"), ("val", "val"), ("test", "test")):
        globals()[f"benchmark_signature_{suffix}"] = benchmark_validation.loc[
            benchmark_validation["signature_split"].eq(split)
        ].reset_index(drop=True)
    run("ad5443e9", "vectorizer = TfidfVectorizer(")
    run("validation-metric")
    run("75c19cc7")
    globals()["bm25_item_matrix"] = make_bm25_matrix(item_count_matrix)
    globals().pop("item_count_matrix")
    run("linear-rerank-setup")
    run("linear-rerank-pool-fn")
    run("linear-rerank-pools")
    run("linear-rerank-features", "overlap_parts, numeric_parts = [], []")

    globals()["rerank_holdout_queries"] = benchmark_signature_test.reset_index(drop=True)
    print("Holdout base pool", flush=True)
    globals()["holdout_candidate_pool"], globals()["holdout_bm25"] = build_rerank_pool(
        rerank_holdout_queries
    )
    globals()["holdout_query_tfidf"] = pair_vectorizer.transform(
        normalize_for_title_match(rerank_holdout_queries["search_query"])
    ).tocsr()
    globals()["val_title_query_matrix"] = pair_vectorizer.transform(
        normalize_for_title_match(rerank_val_queries["search_query"])
    ).tocsr()
    globals()["val_title_candidates"] = get_topk_sparse_predictions(
        rerank_val_queries, rerank_items, val_title_query_matrix, pair_item_matrix,
        top_k=CANDIDATES_PER_CHANNEL, batch_size=32, n_workers=2,
    )
    globals()["holdout_title_query_matrix"] = pair_vectorizer.transform(
        normalize_for_title_match(rerank_holdout_queries["search_query"])
    ).tocsr()
    globals()["holdout_title_candidates"] = get_topk_sparse_predictions(
        rerank_holdout_queries, rerank_items, holdout_title_query_matrix,
        pair_item_matrix, top_k=CANDIDATES_PER_CHANNEL, batch_size=32, n_workers=2,
    )
    # merge_title_rank сам добавит кандидатов title-канала через outer join.
    globals()["val_title_pool"] = val_candidate_pool
    globals()["holdout_title_pool"] = holdout_candidate_pool
    globals()["query_history"] = build_query_history(rerank_train_queries)
    globals()["val_history_predictions"] = get_history_predictions(
        rerank_val_queries, query_history, limit=CANDIDATES_PER_CHANNEL,
    )
    globals()["holdout_history_predictions"] = get_history_predictions(
        rerank_holdout_queries, query_history, limit=CANDIDATES_PER_CHANNEL,
    )
    globals()["HISTORY_LIMIT"] = 20
    # Только для печати сравнения с ранее выполненным фиксированным baseline.
    globals()["rerank_recall"] = 0.52111
    globals()["holdout_recall"] = 0.52168
    globals()["val_history_recall"] = 0.53837
    globals()["holdout_history_recall"] = 0.53912
    run("title_tfidf_reranker_v2")
    if with_dense:
        for name in (
            "title_train_matrix", "train_candidate_pool", "val_candidate_pool",
            "holdout_candidate_pool", "train_title_pool", "val_title_pool",
            "holdout_title_pool", "train_pair_samples", "val_title_candidates",
            "holdout_title_candidates",
        ):
            globals().pop(name, None)
        # Hugging Face использует tqdm.tqdm и в аннотациях типов: возвращаем класс.
        tqdm.tqdm = original_tqdm
        # Переиспользуем блоки ноутбука и те же split/корпус/обученные baseline.
        run("384f0f51")  # get_predictions_sparse для BM25 top-200.
        run("granite-encode")
        run("granite-faiss")
        run("granite-evaluate")
        run("granite-rerank-pool")
        run("granite-rerank-fit")
        run("granite-rerank-evaluate")
    print(f"Elapsed minutes: {(time.time() - started) / 60:.2f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-dense", action="store_true")
    main(with_dense=parser.parse_args().with_dense)
