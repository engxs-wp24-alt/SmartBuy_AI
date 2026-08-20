from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# ============================================================
# File path and required fields
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
PRODUCTS_PATH = PROJECT_DIR / "data" / "products.csv"

REQUIRED_COLUMNS = {
    "product_id",
    "product_name",
    "brand",
    "category",
    "subcategory",
    "price",
    "rating",
    "review_count",
}

DISPLAY_COLUMNS = [
    "product_id",
    "product_name",
    "brand",
    "category",
    "subcategory",
    "price",
    "discount",
    "final_price",
    "rating",
    "review_count",
    "seller_rating",
]


# Search terms commonly used by customers.
SYNONYMS = {
    "phone": "mobile smartphone",
    "phones": "mobile smartphone",
    "smartphone": "mobile smartphone",
    "cellphone": "mobile smartphone",
    "cell phone": "mobile smartphone",
    "notebook": "laptop",
    "computer": "laptop",
    "earphone": "headphones",
    "earphones": "headphones",
    "headset": "headphones",
    "bluetooth": "bluetooth wireless",
    "sport": "sports fitness outdoor cycling",
    "clothes": "clothing wear collection",
    "clothing": "clothing wear collection",
    "fashion": "clothing wear collection",
    "men": "men men's",
    "women": "women women's",
    "child": "kids",
    "children": "kids",
    "blender": "blender kitchen appliance",
    "kitchen": "kitchen appliance home",
    "appliance": "kitchen appliance home",
    "sofa": "furniture home",
    "furniture": "furniture home",
    "decor": "home decor",
    "makeup": "makeup beauty",
    "cosmetic": "makeup beauty",
    "skincare": "skincare beauty",
    "haircare": "haircare beauty",
}


def _normalise_text(value: object) -> str:
    """Normalise customer text while preserving useful apostrophes."""

    text = str(value).lower().strip()
    text = re.sub(r"\bsb[- ]?\d+\b", " ", text)
    text = re.sub(r"\bp\d+\b", " ", text)
    text = re.sub(r"[^a-z0-9'&]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _expand_query(keyword: str) -> str:
    query = _normalise_text(keyword)
    expanded = [query]

    # Expand both complete queries and individual customer terms.
    if query in SYNONYMS:
        expanded.append(SYNONYMS[query])
    for term in query.split():
        if term in SYNONYMS:
            expanded.append(SYNONYMS[term])

    return " ".join(expanded).strip()


def _load_products() -> pd.DataFrame:
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(
            f"products.csv not found: {PRODUCTS_PATH}\n"
            "Run preprocessing/data_cleaning.py first."
        )

    frame = pd.read_csv(PRODUCTS_PATH)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"products.csv is missing columns: {sorted(missing)}")

    if frame["product_id"].duplicated().any():
        raise ValueError("products.csv contains duplicate product IDs.")

    for column in ["product_id", "product_name", "brand", "category", "subcategory"]:
        frame[column] = frame[column].fillna("").astype(str).str.strip()

    for column in ["price", "rating", "review_count"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["search_text"] = (
        frame[["product_name", "brand", "category", "subcategory"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(_normalise_text)
    )
    frame["name_signature"] = frame["product_name"].map(_normalise_text)

    # Repetition gives structured catalog fields sensible importance without
    # creating manual numeric similarity scores.
    frame["content_text"] = (
        frame["product_name"].map(_normalise_text)
        + " "
        + frame["brand"].map(_normalise_text).map(lambda value: f"{value} {value}")
        + " "
        + frame["category"].map(_normalise_text).map(lambda value: f"{value} {value}")
        + " "
        + frame["subcategory"].map(_normalise_text).map(
            lambda value: f"{value} {value} {value}"
        )
    )
    return frame.reset_index(drop=True)


# ============================================================
# Build the TF-IDF model once when this module is imported
# ============================================================

products = _load_products()

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=50_000,
    sublinear_tf=True,
    norm="l2",
)

tfidf_matrix = vectorizer.fit_transform(products["content_text"])

product_index = pd.Series(products.index, index=products["product_id"]).to_dict()


def _output_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in DISPLAY_COLUMNS if column in frame.columns]


def _select_diverse_rows(
    candidates: pd.DataFrame,
    number: int,
    score_column: str,
) -> pd.DataFrame:
    """Select relevant results without repeating the same catalog item type."""

    ordered = candidates.sort_values(
        [score_column, "rating", "review_count"],
        ascending=[False, False, False],
        kind="mergesort",
    )

    selected_indices: list[int] = []
    used_signatures: set[str] = set()
    used_brands: set[str] = set()

    # First pass: different catalog names and different brands.
    for index, row in ordered.iterrows():
        signature = str(row["name_signature"])
        brand = str(row["brand"])
        if signature in used_signatures or brand in used_brands:
            continue
        selected_indices.append(index)
        used_signatures.add(signature)
        used_brands.add(brand)
        if len(selected_indices) == number:
            return ordered.loc[selected_indices].sort_values(
                [score_column, "rating", "review_count"],
                ascending=[False, False, False],
                kind="mergesort",
            )

    # Second pass: brands may repeat, but the catalog name still cannot.
    for index, row in ordered.iterrows():
        if index in selected_indices:
            continue
        signature = str(row["name_signature"])
        if signature in used_signatures:
            continue
        selected_indices.append(index)
        used_signatures.add(signature)
        if len(selected_indices) == number:
            break

    return ordered.loc[selected_indices].sort_values(
        [score_column, "rating", "review_count"],
        ascending=[False, False, False],
        kind="mergesort",
    )


def search_products(
    keyword: str = "",
    category: str | None = None,
    subcategory: str | None = None,
    brand: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_rating: float | None = None,
    limit: int = 20,
) -> pd.DataFrame:
    """Search products and apply optional e-commerce filters."""

    limit = max(1, min(int(limit), 100))
    mask = pd.Series(True, index=products.index)

    if category and category != "All":
        mask &= products["category"].str.casefold().eq(str(category).casefold())
    if subcategory and subcategory != "All":
        mask &= products["subcategory"].str.casefold().eq(str(subcategory).casefold())
    if brand and brand != "All":
        mask &= products["brand"].str.casefold().eq(str(brand).casefold())
    if min_price is not None:
        mask &= products["price"].ge(float(min_price))
    if max_price is not None:
        mask &= products["price"].le(float(max_price))
    if min_rating is not None:
        mask &= products["rating"].ge(float(min_rating))

    candidate_indices = products.index[mask].to_numpy()
    if len(candidate_indices) == 0:
        return products.iloc[0:0][_output_columns(products)].copy()

    query = _expand_query(keyword)
    if not query:
        result = products.loc[candidate_indices].sort_values(
            ["rating", "review_count"], ascending=[False, False]
        )
        return result[_output_columns(result)].head(limit).reset_index(drop=True)

    query_vector = vectorizer.transform([query])
    scores = linear_kernel(query_vector, tfidf_matrix[candidate_indices]).ravel()

    # Give direct title/brand/category matches a small, explainable boost.
    literal_query = _normalise_text(keyword)
    literal_match = (
        products.loc[candidate_indices, "search_text"]
        .str.contains(re.escape(literal_query), case=False, regex=True, na=False)
        .to_numpy(dtype=float)
    )
    scores = scores + (0.15 * literal_match)

    positive = scores > 0
    if not positive.any():
        return products.iloc[0:0][_output_columns(products)].copy()

    candidate_indices = candidate_indices[positive]
    scores = scores[positive]
    candidates = products.loc[candidate_indices].copy()
    candidates["search_score"] = scores
    candidates = _select_diverse_rows(candidates, limit, "search_score")

    result = candidates[_output_columns(candidates)].copy()
    result["search_score"] = candidates["search_score"].round(4).to_numpy()
    return result.reset_index(drop=True)


def recommend_similar(product_id: str, number: int = 5) -> pd.DataFrame:
    """Recommend products most similar to a selected product."""

    product_id = str(product_id).strip()
    if product_id not in product_index:
        raise ValueError(f"Unknown product_id: {product_id}")

    number = max(1, min(int(number), 20))
    selected_index = product_index[product_id]
    selected = products.iloc[selected_index]
    similarities = linear_kernel(
        tfidf_matrix[selected_index], tfidf_matrix
    ).ravel()

    # Similar products must stay in the same product type. Exact duplicate
    # catalog names are excluded even when their product IDs and prices differ.
    candidate_mask = (
        products["subcategory"].eq(selected["subcategory"])
        & products.index.to_series().ne(selected_index)
        & products["name_signature"].ne(selected["name_signature"])
        & (similarities > 0)
    )
    candidates = products.loc[candidate_mask].copy()
    candidates["content_similarity"] = similarities[candidate_mask.to_numpy()]

    selected_price = max(float(selected["price"]), 1.0)
    candidate_prices = candidates["price"].fillna(selected_price).clip(lower=1.0)
    candidates["price_similarity"] = np.minimum(
        candidate_prices, selected_price
    ) / np.maximum(candidate_prices, selected_price)

    selected_rating = float(selected["rating"])
    candidates["rating_similarity"] = (
        1.0 - (candidates["rating"].fillna(selected_rating) - selected_rating).abs() / 4.0
    ).clip(0.0, 1.0)

    candidates["similarity_score"] = (
        0.80 * candidates["content_similarity"]
        + 0.15 * candidates["price_similarity"]
        + 0.05 * candidates["rating_similarity"]
    )

    candidates = _select_diverse_rows(candidates, number, "similarity_score")

    result = candidates[_output_columns(candidates)].copy()
    result["content_similarity"] = candidates["content_similarity"].round(4).to_numpy()
    result["similarity_score"] = candidates["similarity_score"].round(4).to_numpy()

    def reason(row: pd.Series) -> str:
        shared = []
        if row["subcategory"] == selected["subcategory"]:
            shared.append(f"same {row['subcategory']} type")
        if row["brand"] == selected["brand"]:
            shared.append(f"same {row['brand']} brand")
        if not shared and row["category"] == selected["category"]:
            shared.append(f"same {row['category']} category")
        return " and ".join(shared) if shared else "similar product content"

    result["recommendation_reason"] = result.apply(reason, axis=1)
    return result.reset_index(drop=True)


def recommend_by_keyword(keyword: str, number: int = 5) -> pd.DataFrame:
    """Compatibility helper: return top content matches for a keyword."""

    return search_products(keyword=keyword, limit=number)


def get_product(product_id: str) -> pd.Series:
    """Return one product record by ID."""

    product_id = str(product_id).strip()
    if product_id not in product_index:
        raise ValueError(f"Unknown product_id: {product_id}")
    return products.iloc[product_index[product_id]][_output_columns(products)]


if __name__ == "__main__":
    print(f"Products loaded: {len(products):,}")
    print(f"TF-IDF matrix: {tfidf_matrix.shape}")

    print("\nSearch example: laptop")
    print(search_products("laptop", limit=3).to_string(index=False))

    example_id = products.iloc[0]["product_id"]
    print(f"\nSimilar products for {example_id}")
    print(recommend_similar(example_id, number=3).to_string(index=False))
