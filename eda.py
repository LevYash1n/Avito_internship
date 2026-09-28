"""Reproducible exploratory analysis for the Avito internship parquet files.

The script intentionally depends only on packages already declared by the project:
pandas, numpy and pyarrow.  It reads wide text data column-by-column where possible,
so the 490 MB training file does not have to be materialised in memory in full.
"""

from __future__ import annotations

import html
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eda_outputs"
FILES = {
    "benchmark_items": ROOT / "benchmark_items.parquet",
    "benchmark_queries": ROOT / "benchmark_queries.parquet",
    "train": ROOT / "train.parquet",
}

QUERY_COLUMNS = [
    "search_query",
    "search_location_id",
    "search_is_delivery_search",
    "search_infm_params_text",
    "search_category",
]

CATEGORICAL_COLUMNS = {
    "benchmark_items": [
        "item_category_id",
        "item_microcat_id",
        "item_location_id",
        "item_is_phone_hidden",
        "item_is_message_forbidden",
    ],
    "benchmark_queries": [
        "search_category",
        "search_location_id",
        "search_is_delivery_search",
    ],
    "train": [
        "search_category",
        "item_category_id",
        "item_microcat_id",
        "search_is_delivery_search",
        "item_is_phone_hidden",
        "item_is_message_forbidden",
    ],
}

NUMERIC_MEASURE_COLUMNS = {
    "item_price",
    "item_rating",
    "item_rating_reviews_count",
    "item_longitude",
    "item_latitude",
}

TEXT_COLUMNS = {
    "search_query",
    "search_infm_params_text",
    "item_title_raw",
    "item_infm_params_text",
    "item_description_raw",
}

TOKEN_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+", flags=re.UNICODE)


def py(value: Any) -> Any:
    """Convert numpy/pandas values to JSON-friendly Python values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(v) for v in value]
    return value


def pct(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def human_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def quantile_summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p01": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    qs = np.quantile(finite, [0.01, 0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    return py(
        {
            "count": finite.size,
            "mean": finite.mean(),
            "std": finite.std(ddof=1) if finite.size > 1 else 0.0,
            "min": finite.min(),
            "p01": qs[0],
            "p05": qs[1],
            "p25": qs[2],
            "median": qs[3],
            "p75": qs[4],
            "p90": qs[5],
            "p95": qs[6],
            "p99": qs[7],
            "max": finite.max(),
        }
    )


def arrow_to_float(array: pa.ChunkedArray) -> np.ndarray:
    casted = pc.cast(array, pa.float64(), safe=False)
    return casted.to_pandas().to_numpy(dtype=np.float64, na_value=np.nan)


def top_values(array: pa.ChunkedArray, limit: int = 10) -> list[dict[str, Any]]:
    counts = pc.value_counts(array).to_pylist()
    counts.sort(key=lambda row: row["counts"], reverse=True)
    return [
        {"value": "<NA>" if row["values"] is None else str(row["values"]), "count": int(row["counts"])}
        for row in counts[:limit]
    ]


def profile_file(name: str, path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    row_count = parquet.metadata.num_rows
    result: dict[str, Any] = {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "file_size": human_bytes(path.stat().st_size),
        "rows": row_count,
        "columns": parquet.metadata.num_columns,
        "row_groups": parquet.metadata.num_row_groups,
        "schema": [],
        "columns_profile": [],
        "numeric": {},
        "text": {},
        "top_values": {},
    }

    for field in parquet.schema_arrow:
        column = field.name
        array = pq.read_table(path, columns=[column]).column(0)
        null_count = int(array.null_count)
        nan_count = 0
        if pa.types.is_floating(array.type):
            nan_count = int(pc.sum(pc.fill_null(pc.is_nan(array), False)).as_py() or 0)
        missing_count = null_count + nan_count
        distinct = int(pc.count_distinct(array, mode="only_valid").as_py())
        empty_count = None

        result["schema"].append({"column": column, "dtype": str(field.type)})

        if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
            trimmed = pc.utf8_trim_whitespace(array)
            empty_count = int(pc.sum(pc.fill_null(pc.equal(trimmed, ""), False)).as_py() or 0)
            lengths = pc.utf8_length(array).to_pandas().to_numpy(dtype=np.float64, na_value=np.nan)
            length_stats = quantile_summary(lengths)
            length_stats["empty_or_whitespace"] = empty_count
            result["text"][column] = length_stats

        profile_row = {
            "column": column,
            "dtype": str(field.type),
            "missing_count": missing_count,
            "missing_pct": pct(missing_count, row_count),
            "distinct_non_null": distinct,
            "distinct_pct": pct(distinct, row_count),
            "empty_or_whitespace": empty_count,
        }
        result["columns_profile"].append(profile_row)

        if column in NUMERIC_MEASURE_COLUMNS:
            values = arrow_to_float(array)
            stats = quantile_summary(values)
            finite = values[np.isfinite(values)]
            stats.update(
                {
                    "zero_count": int(np.sum(finite == 0)),
                    "negative_count": int(np.sum(finite < 0)),
                }
            )
            result["numeric"][column] = stats

        if column in CATEGORICAL_COLUMNS[name]:
            result["top_values"][column] = top_values(array)

        del array

    return result


def distribution_summary(values: pd.Series) -> dict[str, Any]:
    counts = values.value_counts(dropna=False).to_numpy(dtype=np.float64)
    return {
        "groups": int(counts.size),
        "observations": int(counts.sum()),
        **quantile_summary(counts),
        "singletons": int(np.sum(counts == 1)),
        "singletons_pct": pct(np.sum(counts == 1), counts.size),
    }


def normalize_query(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).lower().split())


def tokenize(value: Any) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return set(TOKEN_RE.findall(str(value).lower()))


def query_title_overlap(queries: pd.Series, titles: pd.Series, sample_size: int = 100_000) -> dict[str, Any]:
    n = len(queries)
    if n > sample_size:
        rng = np.random.default_rng(42)
        positions = np.sort(rng.choice(n, size=sample_size, replace=False))
    else:
        positions = np.arange(n)

    recalls: list[float] = []
    jaccards: list[float] = []
    any_overlap = 0
    full_query_in_title = 0
    valid = 0

    q_values = queries.iloc[positions].tolist()
    t_values = titles.iloc[positions].tolist()
    for query, title in zip(q_values, t_values):
        q_norm = normalize_query(query)
        t_norm = normalize_query(title)
        q_tokens = tokenize(query)
        t_tokens = tokenize(title)
        if not q_tokens:
            continue
        valid += 1
        intersection = q_tokens & t_tokens
        union = q_tokens | t_tokens
        any_overlap += bool(intersection)
        full_query_in_title += bool(q_norm and q_norm in t_norm)
        recalls.append(len(intersection) / len(q_tokens))
        jaccards.append(len(intersection) / len(union) if union else 0.0)

    return {
        "sample_size": int(len(positions)),
        "valid_nonempty_queries": valid,
        "any_token_overlap_count": any_overlap,
        "any_token_overlap_pct": pct(any_overlap, valid),
        "full_query_substring_count": full_query_in_title,
        "full_query_substring_pct": pct(full_query_in_title, valid),
        "query_token_recall": quantile_summary(np.asarray(recalls)),
        "token_jaccard": quantile_summary(np.asarray(jaccards)),
    }


def query_tokens(series: pd.Series, limit: int = 25) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for value in series.dropna():
        counter.update(TOKEN_RE.findall(str(value).lower()))
    return [{"token": token, "count": count} for token, count in counter.most_common(limit)]


def compare_values(left: pd.Series, right: pd.Series, numeric: bool = False) -> np.ndarray:
    both_missing = left.isna().to_numpy() & right.isna().to_numpy()
    if numeric:
        left_num = pd.to_numeric(left, errors="coerce").to_numpy(dtype=np.float64)
        right_num = pd.to_numeric(right, errors="coerce").to_numpy(dtype=np.float64)
        equal = np.isclose(left_num, right_num, rtol=1e-9, atol=1e-9, equal_nan=True)
    else:
        equal = left.astype("string").fillna("<NA>").eq(right.astype("string").fillna("<NA>")).to_numpy()
    return equal | both_missing


def analyse_relations(profiles: dict[str, Any]) -> dict[str, Any]:
    train_core_columns = QUERY_COLUMNS + [
        "item_id",
        "item_location_id",
        "item_category_id",
        "item_microcat_id",
    ]
    train = pd.read_parquet(FILES["train"], columns=train_core_columns)
    benchmark_queries = pd.read_parquet(FILES["benchmark_queries"])
    benchmark_items = pd.read_parquet(
        FILES["benchmark_items"],
        columns=["item_id", "item_location_id", "item_category_id", "item_microcat_id"],
    )

    query_hash = pd.util.hash_pandas_object(train[QUERY_COLUMNS], index=False).astype("uint64")
    query_item_frame = pd.DataFrame({"query_hash": query_hash, "item_id": train["item_id"]})
    query_item_hash = pd.util.hash_pandas_object(query_item_frame, index=False).astype("uint64")

    benchmark_query_hash = pd.util.hash_pandas_object(
        benchmark_queries[QUERY_COLUMNS], index=False
    ).astype("uint64")
    train_query_hashes = set(query_hash.unique().tolist())

    train_item_ids = set(train["item_id"].dropna().tolist())
    benchmark_item_ids = set(benchmark_items["item_id"].dropna().tolist())
    item_overlap = train_item_ids & benchmark_item_ids

    train_norm_queries = train["search_query"].map(normalize_query)
    benchmark_norm_queries = benchmark_queries["search_query"].map(normalize_query)
    train_norm_set = set(train_norm_queries[train_norm_queries.ne("")].unique().tolist())
    text_overlap_mask = benchmark_norm_queries.isin(train_norm_set)
    signature_overlap_mask = benchmark_query_hash.isin(train_query_hashes)

    category_match = train["search_category"].eq(train["item_category_id"])
    location_match = train["search_location_id"].eq(train["item_location_id"])

    pair_counts = query_item_hash.value_counts()
    q_counts = query_hash.value_counts()
    item_counts = train["item_id"].value_counts(dropna=False)

    titles = pd.read_parquet(FILES["train"], columns=["item_title_raw"])["item_title_raw"]
    lexical = query_title_overlap(train["search_query"], titles)

    result: dict[str, Any] = {
        "train": {
            "rows": int(len(train)),
            "unique_query_text_raw": int(train["search_query"].nunique(dropna=False)),
            "unique_query_text_normalized": int(train_norm_queries.nunique(dropna=False)),
            "unique_query_signatures": int(query_hash.nunique(dropna=False)),
            "unique_items": int(train["item_id"].nunique(dropna=False)),
            "unique_query_item_pairs": int(query_item_hash.nunique(dropna=False)),
            "duplicate_query_item_observations": int(pair_counts.sub(1).clip(lower=0).sum()),
            "duplicate_query_item_observations_pct": pct(pair_counts.sub(1).clip(lower=0).sum(), len(train)),
            "repeated_pair_groups": int((pair_counts > 1).sum()),
            "category_match_count": int(category_match.sum()),
            "category_match_pct": pct(category_match.sum(), len(train)),
            "category_mismatch_count": int((~category_match).sum()),
            "location_match_count": int(location_match.sum()),
            "location_match_pct": pct(location_match.sum(), len(train)),
            "location_mismatch_count": int((~location_match).sum()),
            "items_per_query_signature": distribution_summary(query_hash),
            "observations_per_item": distribution_summary(train["item_id"]),
            "observations_per_query_item_pair": distribution_summary(query_item_hash),
            "query_title_lexical_overlap": lexical,
            "top_query_tokens_weighted": query_tokens(train["search_query"]),
        },
        "benchmark": {
            "query_id_unique": int(benchmark_queries["query_id"].nunique(dropna=False)),
            "query_id_duplicates": int(benchmark_queries["query_id"].duplicated().sum()),
            "query_signature_unique": int(benchmark_query_hash.nunique(dropna=False)),
            "item_id_unique": int(benchmark_items["item_id"].nunique(dropna=False)),
            "item_id_duplicates": int(benchmark_items["item_id"].duplicated().sum()),
        },
        "overlap": {
            "benchmark_items_in_train": len(item_overlap),
            "benchmark_items_in_train_pct": pct(len(item_overlap), len(benchmark_item_ids)),
            "train_items_in_benchmark_pct": pct(len(item_overlap), len(train_item_ids)),
            "benchmark_query_signatures_in_train": int(signature_overlap_mask.sum()),
            "benchmark_query_signatures_in_train_pct": pct(signature_overlap_mask.sum(), len(benchmark_queries)),
            "benchmark_normalized_query_texts_in_train": int(text_overlap_mask.sum()),
            "benchmark_normalized_query_texts_in_train_pct": pct(text_overlap_mask.sum(), len(benchmark_queries)),
        },
        "categorical_shift": {
            "train_search_category": category_slice(train["search_category"]),
            "benchmark_search_category": category_slice(benchmark_queries["search_category"]),
            "train_item_category": category_slice(train["item_category_id"]),
            "benchmark_item_category": category_slice(benchmark_items["item_category_id"]),
            "train_delivery_search": binary_slice(train["search_is_delivery_search"]),
            "benchmark_delivery_search": binary_slice(
                benchmark_queries["search_is_delivery_search"]
            ),
            "train_search_locations": int(train["search_location_id"].nunique(dropna=False)),
            "benchmark_search_locations": int(
                benchmark_queries["search_location_id"].nunique(dropna=False)
            ),
            "train_item_microcategories": int(train["item_microcat_id"].nunique(dropna=False)),
            "benchmark_item_microcategories": int(
                benchmark_items["item_microcat_id"].nunique(dropna=False)
            ),
        },
    }

    del titles

    selected_item_columns = [
        "item_id",
        "item_title_raw",
        "item_price",
        "item_rating_reviews_count",
        "item_rating",
        "item_microcat_id",
        "item_location_id",
        "item_is_phone_hidden",
        "item_is_message_forbidden",
        "item_category_id",
    ]
    train_items = pd.read_parquet(FILES["train"], columns=selected_item_columns)
    benchmark_item_attrs = pd.read_parquet(FILES["benchmark_items"], columns=selected_item_columns)

    conflicts: dict[str, Any] = {}
    grouped = train_items.groupby("item_id", dropna=False, sort=False)
    for column in selected_item_columns[1:]:
        counts = grouped[column].nunique(dropna=False)
        conflicts[column] = {
            "items_with_multiple_values": int((counts > 1).sum()),
            "pct_of_train_items": pct((counts > 1).sum(), len(counts)),
            "max_values_per_item": int(counts.max()),
        }

    train_first = train_items.drop_duplicates("item_id", keep="first")
    overlap_frame = train_first.merge(
        benchmark_item_attrs,
        on="item_id",
        how="inner",
        suffixes=("_train", "_benchmark"),
        validate="one_to_one",
    )
    agreement: dict[str, Any] = {}
    numeric_columns = {"item_price", "item_rating_reviews_count", "item_rating"}
    for column in selected_item_columns[1:]:
        equal = compare_values(
            overlap_frame[f"{column}_train"],
            overlap_frame[f"{column}_benchmark"],
            numeric=column in numeric_columns,
        )
        agreement[column] = {
            "equal_count": int(equal.sum()),
            "equal_pct": pct(equal.sum(), len(equal)),
            "different_count": int((~equal).sum()),
        }

    result["item_consistency"] = {
        "train_internal_conflicts": conflicts,
        "overlapping_items_compared": int(len(overlap_frame)),
        "train_vs_benchmark_agreement": agreement,
    }

    result["anomalies"] = {
        "train_price": price_quality(FILES["train"]),
        "benchmark_price": price_quality(FILES["benchmark_items"]),
        "train_rating_missingness": rating_missingness(FILES["train"]),
        "benchmark_rating_missingness": rating_missingness(FILES["benchmark_items"]),
        "train_rating_outside_0_5": numeric_outside_count(FILES["train"], "item_rating", 0, 5),
        "benchmark_rating_outside_0_5": numeric_outside_count(
            FILES["benchmark_items"], "item_rating", 0, 5
        ),
        "train_coordinates": coordinate_quality(FILES["train"]),
        "benchmark_coordinates": coordinate_quality(FILES["benchmark_items"]),
    }
    return result


def category_slice(series: pd.Series) -> dict[str, Any]:
    total = len(series)
    category_114 = int(series.eq(114).sum())
    category_0 = int(series.eq(0).sum())
    return {
        "total": total,
        "category_114_count": category_114,
        "category_114_pct": pct(category_114, total),
        "category_0_count": category_0,
        "category_0_pct": pct(category_0, total),
        "other_count": total - category_114 - category_0,
        "other_pct": pct(total - category_114 - category_0, total),
        "unique": int(series.nunique(dropna=False)),
    }


def binary_slice(series: pd.Series) -> dict[str, Any]:
    total = len(series)
    zeros = int(series.eq(0).sum())
    ones = int(series.eq(1).sum())
    return {
        "total": total,
        "zero_count": zeros,
        "zero_pct": pct(zeros, total),
        "one_count": ones,
        "one_pct": pct(ones, total),
        "other_count": total - zeros - ones,
    }


def numeric_outside_count(path: Path, column: str, low: float, high: float) -> int:
    values = arrow_to_float(pq.read_table(path, columns=[column]).column(0))
    finite = values[np.isfinite(values)]
    return int(np.sum((finite < low) | (finite > high)))


def price_quality(path: Path) -> dict[str, Any]:
    values = arrow_to_float(pq.read_table(path, columns=["item_price"]).column(0))
    finite = values[np.isfinite(values)]
    return {
        "minus_one_count": int(np.sum(finite == -1)),
        "zero_count": int(np.sum(finite == 0)),
        "nonpositive_count": int(np.sum(finite <= 0)),
        "over_1m_count": int(np.sum(finite > 1_000_000)),
        "over_1b_count": int(np.sum(finite > 1_000_000_000)),
    }


def rating_missingness(path: Path) -> dict[str, Any]:
    frame = pd.read_parquet(path, columns=["item_rating", "item_rating_reviews_count"])
    rating_missing = frame["item_rating"].isna()
    reviews_missing = frame["item_rating_reviews_count"].isna()
    return {
        "rating_missing": int(rating_missing.sum()),
        "reviews_missing": int(reviews_missing.sum()),
        "both_missing": int((rating_missing & reviews_missing).sum()),
        "rating_only_missing": int((rating_missing & ~reviews_missing).sum()),
        "reviews_only_missing": int((~rating_missing & reviews_missing).sum()),
    }


def coordinate_quality(path: Path) -> dict[str, Any]:
    latitude = arrow_to_float(pq.read_table(path, columns=["item_latitude"]).column(0))
    longitude = arrow_to_float(pq.read_table(path, columns=["item_longitude"]).column(0))
    present = np.isfinite(latitude) & np.isfinite(longitude)
    invalid = present & ((latitude < -90) | (latitude > 90) | (longitude < -180) | (longitude > 180))
    return {
        "outside_world_bounds": int(invalid.sum()),
        "both_zero": int(np.sum(present & (latitude == 0) & (longitude == 0))),
        "either_missing": int(np.sum(~present)),
    }


def fmt_number(value: Any, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}".replace(",", " ")
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value:,.0f}".replace(",", " ")
    if abs(value) >= 1000:
        return f"{value:,.1f}".replace(",", " ")
    return f"{value:.{digits}f}"


def md_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(map(clean, headers)) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(clean(value) for value in row) + " |")
    return "\n".join(lines)


def html_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def horizontal_bar_svg(
    path: Path,
    title: str,
    labels: list[str],
    values: list[float],
    *,
    suffix: str = "",
    log_scale: bool = False,
) -> None:
    width = 980
    left = 330
    right = 130
    top = 78
    row_h = 34
    height = top + row_h * len(labels) + 40
    scaled = [math.log1p(max(value, 0)) if log_scale else max(value, 0) for value in values]
    maximum = max(scaled) if scaled else 1
    maximum = maximum or 1
    chart_w = width - left - right
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' rx='18' fill='#101827'/>",
        f"<text x='28' y='42' fill='#f8fafc' font-family='Segoe UI,Arial' font-size='23' font-weight='700'>{html.escape(title)}</text>",
    ]
    for i, (label, value, scale_value) in enumerate(zip(labels, values, scaled)):
        y = top + i * row_h
        bar_w = chart_w * scale_value / maximum
        parts.extend(
            [
                f"<text x='{left - 14}' y='{y + 17}' text-anchor='end' fill='#cbd5e1' font-family='Segoe UI,Arial' font-size='14'>{html.escape(label[:42])}</text>",
                f"<rect x='{left}' y='{y}' width='{chart_w}' height='22' rx='6' fill='#243247'/>",
                f"<rect x='{left}' y='{y}' width='{bar_w:.1f}' height='22' rx='6' fill='#38bdf8'/>",
                f"<text x='{left + chart_w + 12}' y='{y + 17}' fill='#e2e8f0' font-family='Segoe UI,Arial' font-size='14'>{html.escape(fmt_number(value))}{html.escape(suffix)}</text>",
            ]
        )
    if log_scale:
        parts.append(
            f"<text x='{left}' y='{height - 12}' fill='#94a3b8' font-family='Segoe UI,Arial' font-size='12'>Длина столбцов показана в логарифмической шкале</text>"
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def create_charts(profiles: dict[str, Any], relations: dict[str, Any]) -> None:
    missing_rows: list[tuple[str, float]] = []
    for dataset, profile in profiles.items():
        for column in profile["columns_profile"]:
            if column["missing_pct"] > 0:
                missing_rows.append((f"{dataset}.{column['column']}", column["missing_pct"]))
    missing_rows.sort(key=lambda row: row[1], reverse=True)
    missing_rows = missing_rows[:15]
    horizontal_bar_svg(
        OUT / "missingness.svg",
        "Наибольшая доля пропусков",
        [row[0] for row in missing_rows],
        [row[1] for row in missing_rows],
        suffix="%",
    )

    overlap = relations["overlap"]
    horizontal_bar_svg(
        OUT / "benchmark_overlap.svg",
        "Пересечение benchmark с train",
        [
            "Объявления benchmark: тот же item_id",
            "Запросы benchmark: полная сигнатура",
            "Запросы benchmark: нормализованный текст",
        ],
        [
            overlap["benchmark_items_in_train_pct"],
            overlap["benchmark_query_signatures_in_train_pct"],
            overlap["benchmark_normalized_query_texts_in_train_pct"],
        ],
        suffix="%",
    )

    train = relations["train"]
    qdist = train["items_per_query_signature"]
    idist = train["observations_per_item"]
    labels = ["p50", "p75", "p90", "p95", "p99", "max"]
    horizontal_bar_svg(
        OUT / "frequency_distribution.svg",
        "Частоты объектов в train",
        [f"Объявлений на запрос — {label}" for label in labels]
        + [f"Наблюдений на item_id — {label}" for label in labels],
        [qdist[key] for key in ["median", "p75", "p90", "p95", "p99", "max"]]
        + [idist[key] for key in ["median", "p75", "p90", "p95", "p99", "max"]],
        log_scale=True,
    )

    lexical = train["query_title_lexical_overlap"]
    horizontal_bar_svg(
        OUT / "text_overlap.svg",
        "Лексическое пересечение запроса и заголовка",
        [
            "Есть хотя бы один общий токен",
            "Запрос целиком входит в заголовок",
            "Медианный recall токенов запроса",
            "Медианный Jaccard токенов",
        ],
        [
            lexical["any_token_overlap_pct"],
            lexical["full_query_substring_pct"],
            lexical["query_token_recall"]["median"] * 100,
            lexical["token_jaccard"]["median"] * 100,
        ],
        suffix="%",
    )


def build_report(profiles: dict[str, Any], relations: dict[str, Any]) -> str:
    train = relations["train"]
    overlap = relations["overlap"]
    anomalies = relations["anomalies"]
    shift = relations["categorical_shift"]
    lexical = train["query_title_lexical_overlap"]

    overview_rows = [
        [name, fmt_number(profile["rows"]), profile["columns"], profile["file_size"], profile["row_groups"]]
        for name, profile in profiles.items()
    ]

    missing_rows: list[list[Any]] = []
    for dataset, profile in profiles.items():
        for row in profile["columns_profile"]:
            if row["missing_count"] or row["empty_or_whitespace"]:
                missing_rows.append(
                    [
                        dataset,
                        row["column"],
                        fmt_number(row["missing_count"]),
                        f"{row['missing_pct']:.2f}%",
                        "—" if row["empty_or_whitespace"] is None else fmt_number(row["empty_or_whitespace"]),
                        fmt_number(row["distinct_non_null"]),
                    ]
                )
    missing_rows.sort(key=lambda row: float(str(row[3]).replace("%", "")), reverse=True)

    numeric_rows: list[list[Any]] = []
    for dataset in ["train", "benchmark_items"]:
        for column, stats in profiles[dataset]["numeric"].items():
            numeric_rows.append(
                [
                    dataset,
                    column,
                    fmt_number(stats["count"]),
                    fmt_number(stats["mean"]),
                    fmt_number(stats["median"]),
                    fmt_number(stats["p95"]),
                    fmt_number(stats["p99"]),
                    fmt_number(stats["max"]),
                ]
            )

    text_rows: list[list[Any]] = []
    for dataset, profile in profiles.items():
        for column, stats in profile["text"].items():
            text_rows.append(
                [
                    dataset,
                    column,
                    fmt_number(stats["count"]),
                    fmt_number(stats["median"]),
                    fmt_number(stats["p95"]),
                    fmt_number(stats["p99"]),
                    fmt_number(stats["max"]),
                    fmt_number(stats["empty_or_whitespace"]),
                ]
            )

    conflict_rows = [
        [
            column,
            fmt_number(values["items_with_multiple_values"]),
            f"{values['pct_of_train_items']:.3f}%",
            values["max_values_per_item"],
        ]
        for column, values in relations["item_consistency"]["train_internal_conflicts"].items()
    ]
    agreement_rows = [
        [
            column,
            fmt_number(values["equal_count"]),
            f"{values['equal_pct']:.2f}%",
            fmt_number(values["different_count"]),
        ]
        for column, values in relations["item_consistency"]["train_vs_benchmark_agreement"].items()
    ]

    categorical_rows = [
        [
            "train.search_category",
            fmt_number(shift["train_search_category"]["category_114_count"]),
            f"{shift['train_search_category']['category_114_pct']:.3f}%",
            fmt_number(shift["train_search_category"]["category_0_count"]),
            f"{shift['train_search_category']['category_0_pct']:.3f}%",
            fmt_number(shift["train_search_category"]["other_count"]),
        ],
        [
            "benchmark_queries.search_category",
            fmt_number(shift["benchmark_search_category"]["category_114_count"]),
            f"{shift['benchmark_search_category']['category_114_pct']:.2f}%",
            fmt_number(shift["benchmark_search_category"]["category_0_count"]),
            f"{shift['benchmark_search_category']['category_0_pct']:.2f}%",
            fmt_number(shift["benchmark_search_category"]["other_count"]),
        ],
        [
            "train.item_category_id",
            fmt_number(shift["train_item_category"]["category_114_count"]),
            f"{shift['train_item_category']['category_114_pct']:.3f}%",
            fmt_number(shift["train_item_category"]["category_0_count"]),
            f"{shift['train_item_category']['category_0_pct']:.3f}%",
            fmt_number(shift["train_item_category"]["other_count"]),
        ],
        [
            "benchmark_items.item_category_id",
            fmt_number(shift["benchmark_item_category"]["category_114_count"]),
            f"{shift['benchmark_item_category']['category_114_pct']:.2f}%",
            fmt_number(shift["benchmark_item_category"]["category_0_count"]),
            f"{shift['benchmark_item_category']['category_0_pct']:.3f}%",
            fmt_number(shift["benchmark_item_category"]["other_count"]),
        ],
    ]

    category_exceptions = train["category_mismatch_count"]
    report = f"""# EDA: Avito query–item data

Дата расчёта: {datetime.now().strftime('%Y-%m-%d %H:%M')}  
Источник: `benchmark_items.parquet`, `benchmark_queries.parquet`, `train.parquet`.

## Краткий итог

- В данных **нет целевой переменной или оценки релевантности**. `train` содержит {fmt_number(train['rows'])} наблюдений пар «сигнатура запроса — item_id» и выглядит как журнал положительных/наблюдавшихся взаимодействий. По этим файлам нельзя оценить баланс классов или построить честную supervised-валидацию без определения отрицательных примеров и метрики.
- В `train` {fmt_number(train['unique_query_signatures'])} уникальных сигнатур запроса и {fmt_number(train['unique_items'])} уникальных объявлений. {fmt_number(train['duplicate_query_item_observations'])} строк ({train['duplicate_query_item_observations_pct']:.2f}%) являются дополнительными наблюдениями уже встречавшейся пары.
- Категория запроса совпадает с категорией объявления в {train['category_match_pct']:.3f}% строк; исключений всего {fmt_number(category_exceptions)}. Локация совпадает в {train['location_match_pct']:.2f}% строк — географическое несовпадение нельзя автоматически считать ошибкой.
- Пересечение benchmark с train неполное: {fmt_number(overlap['benchmark_items_in_train'])} item_id ({overlap['benchmark_items_in_train_pct']:.2f}% benchmark items), {fmt_number(overlap['benchmark_query_signatures_in_train'])} полных сигнатур запросов ({overlap['benchmark_query_signatures_in_train_pct']:.2f}%), но {fmt_number(overlap['benchmark_normalized_query_texts_in_train'])} текстов запросов ({overlap['benchmark_normalized_query_texts_in_train_pct']:.2f}%). Это сочетание transductive/cold-start сценариев.
- Есть сильный categorical shift: `search_category=0` составляет только {shift['train_search_category']['category_0_pct']:.3f}% train, но {shift['benchmark_search_category']['category_0_pct']:.2f}% benchmark queries. Для этой группы итоговую метрику нужно считать отдельно.
- На выборке из {fmt_number(lexical['sample_size'])} train-пар хотя бы один токен запроса встречается в заголовке у {lexical['any_token_overlap_pct']:.2f}% пар; запрос целиком входит в заголовок у {lexical['full_query_substring_pct']:.2f}%. Одного lexical matching будет недостаточно.
- Пропуски рейтинга/числа отзывов вложены: во всех строках без числа отзывов также отсутствует рейтинг. Их нужно кодировать отдельным индикатором, а не только заполнять средним. Не найдено координат вне мирового диапазона; рейтинги лежат в [0, 5].

## 1. Объём и схема

{md_table(['Таблица', 'Строк', 'Колонок', 'Размер', 'Row groups'], overview_rows)}

Все три файла состоят из одного row group. Для итеративной обработки это неудобно: чтение по батчам возможно, но predicate pushdown почти не помогает. Для обучения разумно один раз переписать подготовленные признаки в несколько row groups/партиций.

## 2. Пропуски и кардинальность

![Пропуски](eda_outputs/missingness.svg)

Ниже показаны только колонки, где есть `null`/`NaN` или пустые строки. Пустая строка не считается пропуском и приведена отдельно.

{md_table(['Таблица', 'Колонка', 'Null/NaN', 'Доля', 'Пустые строки', 'Уникальных non-null'], missing_rows)}

Практический вывод: для текстовых полей следует различать `null` и пустую строку только если это имеет смысл в источнике; для модели их можно привести к единому специальному токену. Для рейтингов полезны пары признаков `rating_filled` + `rating_is_missing`.

## 3. Числовые признаки

{md_table(['Таблица', 'Признак', 'N', 'Среднее', 'Медиана', 'p95', 'p99', 'Максимум'], numeric_rows)}

- Цена и число отзывов имеют длинный правый хвост. Средняя цена здесь неустойчива: максимум почти `10^12`, а значений выше 1 млрд — train {fmt_number(anomalies['train_price']['over_1b_count'])}, benchmark {fmt_number(anomalies['benchmark_price']['over_1b_count'])}. Для модели нужны отдельный флаг специальных значений, `log1p` только для положительных цен и clip по train-квантилям.
- `item_price=-1` явно выглядит как sentinel: train — {fmt_number(anomalies['train_price']['minus_one_count'])}, benchmark — {fmt_number(anomalies['benchmark_price']['minus_one_count'])}. Нулей отдельно: {fmt_number(anomalies['train_price']['zero_count'])} и {fmt_number(anomalies['benchmark_price']['zero_count'])}; их нельзя смешивать с настоящей ценой до уточнения семантики.
- Рейтингов вне [0, 5]: train — {fmt_number(anomalies['train_rating_outside_0_5'])}, benchmark — {fmt_number(anomalies['benchmark_rating_outside_0_5'])}. Координат вне мировых границ нет, но пар `(0, 0)` — train {fmt_number(anomalies['train_coordinates']['both_zero'])}, benchmark {fmt_number(anomalies['benchmark_coordinates']['both_zero'])}; это вероятный sentinel.

### 3.1. Категориальный сдвиг train → benchmark

{md_table(['Срез', 'category=114', 'Доля 114', 'category=0', 'Доля 0', 'Другие'], categorical_rows)}

- `search_category=0`, вероятно, означает запрос без выбранной категории. Таких benchmark-запросов {fmt_number(shift['benchmark_search_category']['category_0_count'])}, а в train всего {fmt_number(shift['train_search_category']['category_0_count'])}: это наиболее явный out-of-distribution сегмент.
- `search_is_delivery_search=1` встречается в train лишь {fmt_number(shift['train_delivery_search']['one_count'])} раз, а в benchmark отсутствует полностью. Этот флаг практически не обучаем и не проверяется текущим benchmark.
- Уникальных search locations: train {fmt_number(shift['train_search_locations'])}, benchmark {fmt_number(shift['benchmark_search_locations'])}; item microcategories: {fmt_number(shift['train_item_microcategories'])} и {fmt_number(shift['benchmark_item_microcategories'])} соответственно.

## 4. Текстовые поля

Длины указаны в Unicode-символах, не в токенах модели.

{md_table(['Таблица', 'Поле', 'N', 'Медиана', 'p95', 'p99', 'Максимум', 'Пустых'], text_rows)}

![Лексическое пересечение](eda_outputs/text_overlap.svg)

- Медианный recall токенов запроса по заголовку: {lexical['query_token_recall']['median']:.3f}; медианный Jaccard: {lexical['token_jaccard']['median']:.3f}.
- Для baseline имеет смысл сравнить BM25/TF-IDF по конкатенации title + attributes + description с bi-encoder. Длинные описания следует обрезать осознанно и измерить вклад отдельно.
- HTML/emoji/телефоноподобные строки и повторяющаяся пунктуация могут быть полезным сигналом качества объявления, поэтому агрессивно очищать текст до букв не стоит.

## 5. Структура train и повторы

{md_table(
    ['Показатель', 'Значение'],
    [
        ['Уникальных сырых текстов запроса', fmt_number(train['unique_query_text_raw'])],
        ['Уникальных нормализованных текстов запроса', fmt_number(train['unique_query_text_normalized'])],
        ['Уникальных полных сигнатур запроса', fmt_number(train['unique_query_signatures'])],
        ['Уникальных item_id', fmt_number(train['unique_items'])],
        ['Уникальных query–item пар', fmt_number(train['unique_query_item_pairs'])],
        ['Дополнительных повторов query–item', fmt_number(train['duplicate_query_item_observations'])],
        ['Групп повторяющихся query–item пар', fmt_number(train['repeated_pair_groups'])],
        ['Совпадение search/item category', f"{train['category_match_pct']:.3f}%"],
        ['Совпадение search/item location', f"{train['location_match_pct']:.2f}%"],
    ],
)}

![Частоты](eda_outputs/frequency_distribution.svg)

Медиана — одно объявление на полную сигнатуру запроса и одно наблюдение на item_id, но максимумы равны соответственно {fmt_number(train['items_per_query_signature']['max'])} и {fmt_number(train['observations_per_item']['max'])}. При случайном разбиении строк один и тот же запрос или item легко попадёт и в train, и в validation, завысив качество.

## 6. Benchmark и риск утечки

![Пересечение benchmark](eda_outputs/benchmark_overlap.svg)

{md_table(
    ['Проверка', 'Количество', 'Доля benchmark'],
    [
        ['item_id встречался в train', fmt_number(overlap['benchmark_items_in_train']), f"{overlap['benchmark_items_in_train_pct']:.2f}%"],
        ['Полная сигнатура запроса встречалась в train', fmt_number(overlap['benchmark_query_signatures_in_train']), f"{overlap['benchmark_query_signatures_in_train_pct']:.2f}%"],
        ['Нормализованный текст запроса встречался в train', fmt_number(overlap['benchmark_normalized_query_texts_in_train']), f"{overlap['benchmark_normalized_query_texts_in_train_pct']:.2f}%"],
    ],
)}

Сам факт пересечения item_id не является ошибкой: benchmark может специально проверять ранжирование известного каталога. Но признаки, агрегированные по `item_id` или тексту запроса, должны строиться **только на train-части каждого фолда**. Нельзя заранее присоединять частоты/CTR, рассчитанные по всему `train`, к validation.

## 7. Стабильность атрибутов объявления

Часть item_id встречается многократно. Проверка ниже показывает, менялось ли значение атрибута внутри train.

{md_table(['Атрибут', 'Item с >1 значением', 'Доля item', 'Максимум значений'], conflict_rows)}

Для {fmt_number(relations['item_consistency']['overlapping_items_compared'])} item_id, общих у train и benchmark, сопоставлены первые наблюдаемые значения:

{md_table(['Атрибут', 'Совпало', 'Доля совпадений', 'Различается'], agreement_rows)}

Все выбранные атрибуты у {fmt_number(relations['item_consistency']['overlapping_items_compared'])} общих item_id совпали на 100%, а внутри train для одного item_id не найдено конфликтующих значений. Это указывает на единый snapshot каталога, а не историю изменений. Поэтому при валидации важно помнить: train-строка не содержит состояние объявления на момент реального взаимодействия — только предоставленный snapshot.

## 8. Рекомендации для моделирования и валидации

1. Сначала уточнить постановку: что считается релевантным, как формируются кандидаты и какая метрика используется (`Recall@K`, `NDCG@K`, `MRR` и т. п.). В текущих файлах нет label и benchmark ground truth.
2. Повторные одинаковые query–item наблюдения либо удалить для бинарной релевантности, либо превратить в вес/счётчик, если частота действительно отражает силу сигнала.
3. Делить данные группами, а не случайными строками. Подготовить как минимум два среза: unseen query signature и unseen item; отдельно измерять warm/cold performance и `search_category=0`.
4. Базовые признаки: нормализованный title/query, BM25/TF-IDF; category/microcategory; точное совпадение location; log-price/log-reviews; missingness рейтинга; флаги связи; длины текстов; token overlap.
5. Для semantic retrieval обучать bi-encoder с in-batch negatives и hard negatives из той же категории/локации. Любые отрицательные примеры нужно формировать так, чтобы необозначенный позитив не превратился автоматически в «истинный негатив».
6. Cross-encoder использовать как reranker ограниченного candidate set. Длинные descriptions и attributes лучше аблировать отдельно: они существенно увеличивают стоимость.
7. Зафиксировать дедупликацию, normalization и snapshot атрибутов до обучения; иначе offline-оценка будет нестабильной и может содержать утечку.

## Ограничения анализа

- Нельзя посчитать качество таргета, дисбаланс классов и корреляции с label: label отсутствует.
- Нет временной метки, поэтому невозможно проверить drift и сделать time split.
- Нет файла соответствий benchmark query–item с релевантностью, поэтому покрытие не равно качеству retrieval.
- Лексический overlap рассчитан на детерминированной выборке до 100 000 строк (`seed=42`); остальные сводки точные для всех строк.
"""
    return report


def build_html(report: str, profiles: dict[str, Any], relations: dict[str, Any]) -> str:
    train = relations["train"]
    overlap = relations["overlap"]
    shift = relations["categorical_shift"]
    anomalies = relations["anomalies"]
    cards = [
        ("Строк train", fmt_number(train["rows"]), "наблюдений"),
        ("Уникальных запросов", fmt_number(train["unique_query_signatures"]), "полных сигнатур"),
        ("Уникальных объявлений", fmt_number(train["unique_items"]), "item_id в train"),
        ("Повторных пар", f"{train['duplicate_query_item_observations_pct']:.2f}%", "query–item observations"),
        ("Benchmark item overlap", f"{overlap['benchmark_items_in_train_pct']:.2f}%", "по item_id"),
        ("Benchmark query overlap", f"{overlap['benchmark_query_signatures_in_train_pct']:.2f}%", "по полной сигнатуре"),
    ]
    card_html = "".join(
        f"<div class='card'><div class='card-label'>{html.escape(label)}</div><div class='card-value'>{html.escape(value)}</div><div class='card-note'>{html.escape(note)}</div></div>"
        for label, value, note in cards
    )

    overview_rows = [
        [name, fmt_number(profile["rows"]), profile["columns"], profile["file_size"]]
        for name, profile in profiles.items()
    ]
    lexical = train["query_title_lexical_overlap"]
    return f"""<!doctype html>
<html lang='ru'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>EDA — Avito query–item data</title>
<style>
:root{{--bg:#08111f;--panel:#101827;--panel2:#172338;--text:#e5edf7;--muted:#94a3b8;--accent:#38bdf8;--accent2:#22c55e;--warn:#f59e0b;}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(135deg,#07101d,#101827 55%,#0b1525);color:var(--text);font:15px/1.55 'Segoe UI',Arial,sans-serif}}
.wrap{{max-width:1180px;margin:0 auto;padding:48px 24px 80px}} h1{{font-size:42px;line-height:1.1;margin:0 0 10px}} h2{{margin-top:42px;font-size:26px}} h3{{font-size:18px}} .subtitle,.muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin:28px 0}} .card{{background:rgba(16,24,39,.92);border:1px solid #26354c;border-radius:15px;padding:18px}} .card-label{{color:var(--muted);font-size:13px}} .card-value{{font-size:27px;font-weight:750;color:#f8fafc;margin:3px 0}} .card-note{{font-size:12px;color:#718096}}
.callout{{background:#13263a;border-left:4px solid var(--accent);border-radius:8px;padding:17px 19px;margin:22px 0}} .warn{{border-left-color:var(--warn)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}} .plot{{width:100%;border:1px solid #26354c;border-radius:15px;background:#101827}}
.table-wrap{{overflow:auto;border:1px solid #26354c;border-radius:12px}} table{{border-collapse:collapse;width:100%;background:#101827}} th,td{{padding:10px 12px;border-bottom:1px solid #26354c;text-align:left;white-space:nowrap}} th{{position:sticky;top:0;background:#172338;color:#bae6fd}} tr:last-child td{{border-bottom:none}}
code{{background:#172338;padding:2px 6px;border-radius:5px}} li{{margin:7px 0}} a{{color:var(--accent)}} footer{{margin-top:48px;color:var(--muted);font-size:13px}}
@media(max-width:620px){{h1{{font-size:31px}}.grid{{grid-template-columns:1fr}}.wrap{{padding:28px 14px}}}}
</style>
</head>
<body><main class='wrap'>
<div class='subtitle'>Exploratory Data Analysis · {datetime.now().strftime('%Y-%m-%d')}</div>
<h1>Avito query–item data</h1>
<p class='subtitle'>Профиль качества, структура взаимодействий и пересечение train/benchmark.</p>
<section class='cards'>{card_html}</section>
<div class='callout warn'><strong>Главное ограничение:</strong> в предоставленных файлах нет label/оценки релевантности и временной метки. Train выглядит как набор наблюдавшихся query–item пар. Баланс классов, drift и честную supervised-валидацию по этим данным определить нельзя.</div>
<h2>Данные</h2>
{html_table(['Таблица','Строк','Колонок','Размер'], overview_rows)}
<h2>Ключевые результаты</h2>
<ul>
<li>{fmt_number(train['duplicate_query_item_observations'])} дополнительных наблюдений уже встречавшейся query–item пары ({train['duplicate_query_item_observations_pct']:.2f}% строк).</li>
<li>Категория query и item совпадает в {train['category_match_pct']:.3f}% строк, location_id — в {train['location_match_pct']:.2f}%.</li>
<li>Benchmark overlap: {overlap['benchmark_items_in_train_pct']:.2f}% item_id, {overlap['benchmark_query_signatures_in_train_pct']:.2f}% полных query signatures, {overlap['benchmark_normalized_query_texts_in_train_pct']:.2f}% нормализованных текстов.</li>
<li><code>search_category=0</code>: {shift['train_search_category']['category_0_pct']:.3f}% в train против {shift['benchmark_search_category']['category_0_pct']:.2f}% в benchmark queries — явный distribution shift.</li>
<li><code>item_price=-1</code>: {fmt_number(anomalies['train_price']['minus_one_count'])} строк train; это специальное значение, которое нельзя подавать модели как обычную отрицательную цену.</li>
<li>В {lexical['any_token_overlap_pct']:.2f}% проверенных пар запрос и title имеют общий токен; медианный query-token recall равен {lexical['query_token_recall']['median']:.3f}.</li>
</ul>
<div class='grid'>
<img class='plot' src='eda_outputs/missingness.svg' alt='Пропуски'>
<img class='plot' src='eda_outputs/benchmark_overlap.svg' alt='Пересечение benchmark'>
<img class='plot' src='eda_outputs/frequency_distribution.svg' alt='Частоты'>
<img class='plot' src='eda_outputs/text_overlap.svg' alt='Пересечение текста'>
</div>
<h2>Что делать дальше</h2>
<ol>
<li>Уточнить схему ground truth, candidate generation и целевую ranking-метрику.</li>
<li>Дедуплицировать пары или превратить число повторов в вес.</li>
<li>Валидироваться по группам query/item; отдельно измерять warm и cold-start.</li>
<li>Начать с BM25/TF-IDF + структурных признаков, затем сравнить bi-encoder и cross-encoder reranking.</li>
<li>Считать любые target/frequency aggregates строго внутри train-фолда.</li>
</ol>
<div class='callout'>Полный разбор, таблицы квантилей, проверка согласованности атрибутов и ограничения находятся в <a href='eda_report.md'>eda_report.md</a>. Машиночитаемые метрики — <a href='eda_outputs/metrics.json'>metrics.json</a>.</div>
<footer>Сформировано скриптом <code>eda.py</code>. Все агрегаты точные, кроме query–title lexical overlap: детерминированная выборка до 100 000 строк, seed=42.</footer>
</main></body></html>"""


def build_notebook(report: str) -> dict[str, Any]:
    def markdown_cell(source: str) -> dict[str, Any]:
        return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}

    def code_cell(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }

    return {
        "cells": [
            markdown_cell(
                "# EDA: Avito query–item data\n\n"
                "Ноутбук является точкой входа к воспроизводимому анализу. "
                "Тяжёлые вычисления вынесены в `eda.py`, чтобы не дублировать логику и обрабатывать текстовые столбцы по одному.\n"
            ),
            code_cell(
                "from pathlib import Path\n"
                "import json\n"
                "import pandas as pd\n"
                "from IPython.display import display, HTML, Markdown\n\n"
                "ROOT = Path.cwd()\n"
                "OUT = ROOT / 'eda_outputs'\n"
            ),
            markdown_cell(
                "## Полный пересчёт\n\n"
                "Следующая ячейка заново профилирует три Parquet-файла и обновляет отчёты. "
                "Для обычного просмотра её запускать не нужно.\n"
            ),
            code_cell("%run eda.py\n"),
            markdown_cell("## Результаты последнего расчёта\n"),
            code_cell(
                "metrics = json.loads((OUT / 'metrics.json').read_text(encoding='utf-8'))\n"
                "display(pd.DataFrame(metrics['profiles']['train']['columns_profile']))\n"
            ),
            markdown_cell(report),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def save_csv_tables(profiles: dict[str, Any], relations: dict[str, Any]) -> None:
    for dataset, profile in profiles.items():
        pd.DataFrame(profile["columns_profile"]).to_csv(
            OUT / f"{dataset}_column_profile.csv", index=False, encoding="utf-8-sig"
        )
    conflict_rows = []
    for column, values in relations["item_consistency"]["train_internal_conflicts"].items():
        conflict_rows.append({"column": column, **values})
    pd.DataFrame(conflict_rows).to_csv(
        OUT / "train_item_attribute_conflicts.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    missing = [str(path) for path in FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required parquet files not found: {missing}")

    OUT.mkdir(exist_ok=True)
    print("1/5 Profiling parquet columns...")
    profiles = {name: profile_file(name, path) for name, path in FILES.items()}
    print("2/5 Analysing relationships, duplicates and overlap...")
    relations = analyse_relations(profiles)
    metrics = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "methodology": {
            "exact_aggregates": True,
            "lexical_overlap_sample_max_rows": 100_000,
            "lexical_overlap_seed": 42,
            "query_signature_columns": QUERY_COLUMNS,
        },
        "profiles": profiles,
        "relations": relations,
    }
    print("3/5 Creating charts...")
    create_charts(profiles, relations)
    print("4/5 Writing reports and tables...")
    report = build_report(profiles, relations)
    (ROOT / "eda_report.md").write_text(report, encoding="utf-8")
    (ROOT / "eda_report.html").write_text(build_html(report, profiles, relations), encoding="utf-8")
    (OUT / "metrics.json").write_text(
        json.dumps(py(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_csv_tables(profiles, relations)
    notebook = build_notebook(report)
    (ROOT / "eda.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("5/5 Done")
    print(f"Markdown: {ROOT / 'eda_report.md'}")
    print(f"HTML:     {ROOT / 'eda_report.html'}")
    print(f"Notebook: {ROOT / 'eda.ipynb'}")


if __name__ == "__main__":
    main()
