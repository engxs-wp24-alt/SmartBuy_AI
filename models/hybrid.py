from __future__ import annotations

import re

import numpy as np
import pandas as pd

try:
    # Normal use: imported from app.py or another project module.
    from models.content_based import (
        products,
        recommend_similar,
        search_products,
    )
    from models.collaborative import (
        get_user_history,
        recommend_for_user,
        users_also_purchased,
    )
except ModuleNotFoundError:
    # Direct test: python models/hybrid.py
    from content_based import products, recommend_similar, search_products
    from collaborative import (
        get_user_history,
        recommend_for_user,
        users_also_purchased,
    )


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

product_lookup = products.set_index("product_id", drop=False)


def _display_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in DISPLAY_COLUMNS if column in frame.columns]


def _normalise_score(values: pd.Series) -> pd.Series:
    """Scale available positive scores to 0-1 while preserving missing zeros."""

    values = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    positive = values > 0
    result = pd.Series(0.0, index=values.index)

    if not positive.any():
        return result

    minimum = values[positive].min()
    maximum = values[positive].max()
    if np.isclose(minimum, maximum):
        result.loc[positive] = 1.0
    else:
        result.loc[positive] = (
            values.loc[positive] - minimum
        ) / (maximum - minimum)
    return result


def _name_signature(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\bsb[- ]?\d+\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _quality_score(frame: pd.DataFrame) -> pd.Series:
    rating = pd.to_numeric(frame["rating"], errors="coerce").fillna(0).clip(0, 5) / 5
    reviews = pd.to_numeric(frame["review_count"], errors="coerce").fillna(0).clip(lower=0)
    popularity = np.log1p(reviews)
    if popularity.max() > 0:
        popularity = popularity / popularity.max()
    return (0.75 * rating + 0.25 * popularity).clip(0, 1)


def _select_diverse(frame: pd.DataFrame, number: int) -> pd.DataFrame:
    """Keep ranking quality while avoiding repeated synthetic catalog names."""

    ordered = frame.sort_values(
        ["hybrid_score", "rating", "review_count"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    ordered = ordered.copy()
    ordered["_signature"] = ordered["product_name"].map(_name_signature)

    selected: list[int] = []
    used_signatures: set[str] = set()
    used_brands: set[str] = set()

    # First pass encourages different brands and different catalog names.
    for index, row in ordered.iterrows():
        signature = str(row["_signature"])
        brand = str(row["brand"])
        if signature in used_signatures or brand in used_brands:
            continue
        selected.append(index)
        used_signatures.add(signature)
        used_brands.add(brand)
        if len(selected) == number:
            break

    # Second pass allows a repeated brand but never an identical catalog name.
    if len(selected) < number:
        for index, row in ordered.iterrows():
            if index in selected:
                continue
            signature = str(row["_signature"])
            if signature in used_signatures:
                continue
            selected.append(index)
            used_signatures.add(signature)
            if len(selected) == number:
                break

    return (
        ordered.loc[selected]
        .sort_values("hybrid_score", ascending=False)
        .drop(columns="_signature")
    )


def _catalog_candidates(product_ids: set[str]) -> pd.DataFrame:
    valid_ids = [product_id for product_id in product_ids if product_id in product_lookup.index]
    if not valid_ids:
        return products.iloc[0:0].copy()
    return product_lookup.loc[valid_ids].copy().reset_index(drop=True)


def hybrid_for_product(
    product_id: str,
    number: int = 5,
    content_weight: float = 0.65,
    collaborative_weight: float = 0.25,
) -> pd.DataFrame:
    """Combine similar products with Users Also Purchased evidence."""

    product_id = str(product_id).strip()
    if product_id not in product_lookup.index:
        raise ValueError(f"Unknown product_id: {product_id}")

    number = max(1, min(int(number), 20))
    selected = product_lookup.loc[product_id]

    content = recommend_similar(product_id, number=20)
    collaborative = users_also_purchased(product_id, number=20)

    content_scores = (
        content.set_index("product_id")["similarity_score"].to_dict()
        if not content.empty else {}
    )
    collaborative_scores = (
        collaborative.set_index("product_id")["collaborative_score"].to_dict()
        if not collaborative.empty else {}
    )
    shared_users = (
        collaborative.set_index("product_id")["shared_user_count"].to_dict()
        if "shared_user_count" in collaborative.columns else {}
    )

    candidate_ids = set(content_scores) | set(collaborative_scores)
    candidate_ids.discard(product_id)
    candidates = _catalog_candidates(candidate_ids)
    candidates = candidates[candidates["category"].eq(selected["category"])].copy()
    if candidates.empty:
        return candidates

    candidates["content_raw"] = candidates["product_id"].map(content_scores).fillna(0.0)
    candidates["collaborative_raw"] = (
        candidates["product_id"].map(collaborative_scores).fillna(0.0)
    )
    candidates["content_score"] = _normalise_score(candidates["content_raw"])
    candidates["collaborative_score"] = _normalise_score(
        candidates["collaborative_raw"]
    )
    candidates["quality_score"] = _quality_score(candidates)

    # A small relevance guard keeps hybrid results connected to the selected
    # product while still allowing genuine cross-category collaborative items.
    candidates["catalog_relevance"] = np.select(
        [
            candidates["subcategory"].eq(selected["subcategory"]),
            candidates["category"].eq(selected["category"]),
        ],
        [1.0, 0.45],
        default=0.0,
    )

    quality_weight = max(0.0, 1.0 - content_weight - collaborative_weight)
    candidates["hybrid_score"] = (
        content_weight * candidates["content_score"]
        + collaborative_weight * candidates["collaborative_score"]
        + quality_weight * candidates["quality_score"]
        + 0.05 * candidates["catalog_relevance"]
    )
    candidates["shared_user_count"] = (
        candidates["product_id"].map(shared_users).fillna(0).astype(int)
    )

    candidates = _select_diverse(candidates, number)
    result = candidates[_display_columns(candidates)].copy()
    result["content_score"] = candidates["content_score"].round(4).to_numpy()
    result["collaborative_score"] = (
        candidates["collaborative_score"].round(4).to_numpy()
    )
    result["shared_user_count"] = candidates["shared_user_count"].to_numpy()
    result["hybrid_score"] = candidates["hybrid_score"].round(4).to_numpy()

    def explanation(row: pd.Series) -> str:
        reasons = []
        if row["content_score"] > 0:
            reasons.append("similar product features")
        if row["shared_user_count"] > 0:
            reasons.append(f"purchased by {int(row['shared_user_count'])} shared user(s)")
        if not reasons:
            reasons.append("rating and popularity")
        return " + ".join(reasons)

    result["recommendation_reason"] = result.apply(explanation, axis=1)
    return result.reset_index(drop=True)


def hybrid_for_user(
    user_id: str,
    number: int = 5,
    collaborative_weight: float = 0.55,
    content_weight: float = 0.35,
) -> pd.DataFrame:
    """Combine collaborative recommendations with a user's content interests."""

    user_id = str(user_id).strip()
    number = max(1, min(int(number), 20))

    history = get_user_history(user_id)
    collaborative = recommend_for_user(user_id, number=20)

    collaborative_scores = (
        collaborative.set_index("product_id")["collaborative_score"].to_dict()
        if not collaborative.empty else {}
    )

    content_scores: dict[str, float] = {}
    supporting_items: dict[str, int] = {}

    # Use up to five strongest preferences to avoid one long history dominating.
    if not history.empty:
        strongest_history = history.sort_values("user_rating", ascending=False).head(5)
        for _, history_item in strongest_history.iterrows():
            similar = recommend_similar(str(history_item["product_id"]), number=10)
            preference = max(float(history_item["user_rating"]) / 5.0, 0.2)
            for _, candidate in similar.iterrows():
                candidate_id = str(candidate["product_id"])
                contribution = float(candidate["similarity_score"]) * preference
                content_scores[candidate_id] = content_scores.get(candidate_id, 0.0) + contribution
                supporting_items[candidate_id] = supporting_items.get(candidate_id, 0) + 1

    candidate_ids = set(collaborative_scores) | set(content_scores)
    seen_ids = set(history["product_id"].astype(str)) if not history.empty else set()
    candidate_ids -= seen_ids

    candidates = _catalog_candidates(candidate_ids)
    if candidates.empty:
        return collaborative.head(number).reset_index(drop=True)

    candidates["collaborative_raw"] = (
        candidates["product_id"].map(collaborative_scores).fillna(0.0)
    )
    candidates["content_raw"] = candidates["product_id"].map(content_scores).fillna(0.0)
    candidates["collaborative_score"] = _normalise_score(
        candidates["collaborative_raw"]
    )
    candidates["content_score"] = _normalise_score(candidates["content_raw"])
    candidates["quality_score"] = _quality_score(candidates)
    candidates["supporting_history_items"] = (
        candidates["product_id"].map(supporting_items).fillna(0).astype(int)
    )

    quality_weight = max(0.0, 1.0 - collaborative_weight - content_weight)
    candidates["hybrid_score"] = (
        collaborative_weight * candidates["collaborative_score"]
        + content_weight * candidates["content_score"]
        + quality_weight * candidates["quality_score"]
    )

    candidates = _select_diverse(candidates, number)
    result = candidates[_display_columns(candidates)].copy()
    result["content_score"] = candidates["content_score"].round(4).to_numpy()
    result["collaborative_score"] = (
        candidates["collaborative_score"].round(4).to_numpy()
    )
    result["supporting_history_items"] = (
        candidates["supporting_history_items"].to_numpy()
    )
    result["hybrid_score"] = candidates["hybrid_score"].round(4).to_numpy()

    def explanation(row: pd.Series) -> str:
        reasons = []
        if row["collaborative_score"] > 0:
            reasons.append("users with related purchase patterns")
        if row["supporting_history_items"] > 0:
            reasons.append(
                f"similar to {int(row['supporting_history_items'])} history item(s)"
            )
        if not reasons:
            reasons.append("popular and highly rated")
        return " + ".join(reasons)

    result["recommendation_reason"] = result.apply(explanation, axis=1)
    return result.reset_index(drop=True)


def hybrid_recommend(
    keyword: str | None = None,
    product_id: str | None = None,
    user_id: str | None = None,
    number: int = 5,
) -> pd.DataFrame:
    """Compatibility entry point for app.py.

    Priority:
    - user_id supplied: personalised hybrid recommendations.
    - product_id supplied: hybrid recommendations for the selected product.
    - keyword supplied: search the keyword, then use its top product.
    """

    if user_id:
        return hybrid_for_user(user_id=user_id, number=number)

    if product_id:
        return hybrid_for_product(product_id=product_id, number=number)

    if keyword:
        matches = search_products(keyword=keyword, limit=1)
        if matches.empty:
            return pd.DataFrame()
        return hybrid_for_product(
            product_id=str(matches.iloc[0]["product_id"]),
            number=number,
        )

    raise ValueError("Provide product_id, user_id, or keyword.")


if __name__ == "__main__":
    example_product = str(products.iloc[0]["product_id"])
    print(f"Hybrid recommendations for product {example_product}")
    print(hybrid_for_product(example_product, number=3).to_string(index=False))

    example_history = get_user_history("U272481")
    if not example_history.empty:
        print("\nPersonalised hybrid recommendations for U272481")
        print(hybrid_for_user("U272481", number=3).to_string(index=False))
