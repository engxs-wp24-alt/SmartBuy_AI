from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ============================================================
# Paths and expected columns
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
RATINGS_PATH = PROJECT_DIR / "data" / "ratings.csv"
PRODUCTS_PATH = PROJECT_DIR / "data" / "products.csv"

RATING_COLUMNS = {"user_id", "product_id", "rating"}
PRODUCT_COLUMNS = {
    "product_id", "product_name", "brand", "category", "subcategory",
    "price", "rating", "review_count",
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


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RATINGS_PATH.exists():
        raise FileNotFoundError(
            f"ratings.csv not found: {RATINGS_PATH}\n"
            "Run preprocessing/data_cleaning.py first."
        )
    if not PRODUCTS_PATH.exists():
        raise FileNotFoundError(
            f"products.csv not found: {PRODUCTS_PATH}\n"
            "Run preprocessing/data_cleaning.py first."
        )

    rating_data = pd.read_csv(RATINGS_PATH)
    product_data = pd.read_csv(PRODUCTS_PATH)

    missing_ratings = RATING_COLUMNS.difference(rating_data.columns)
    missing_products = PRODUCT_COLUMNS.difference(product_data.columns)
    if missing_ratings:
        raise ValueError(f"ratings.csv is missing columns: {sorted(missing_ratings)}")
    if missing_products:
        raise ValueError(f"products.csv is missing columns: {sorted(missing_products)}")

    rating_data = rating_data[["user_id", "product_id", "rating"]].copy()
    rating_data["user_id"] = rating_data["user_id"].astype(str).str.strip()
    rating_data["product_id"] = rating_data["product_id"].astype(str).str.strip()
    rating_data["rating"] = pd.to_numeric(rating_data["rating"], errors="coerce")
    rating_data = rating_data.dropna(subset=["user_id", "product_id", "rating"])
    rating_data = rating_data[rating_data["rating"].between(1, 5)].copy()

    if rating_data.duplicated(["user_id", "product_id"]).any():
        rating_data = (
            rating_data.groupby(["user_id", "product_id"], as_index=False)
            .agg(rating=("rating", "mean"))
        )

    product_data["product_id"] = product_data["product_id"].astype(str).str.strip()
    product_data = product_data.drop_duplicates("product_id").reset_index(drop=True)

    known_products = set(product_data["product_id"])
    rating_data = rating_data[rating_data["product_id"].isin(known_products)].copy()
    if rating_data.empty:
        raise ValueError("No valid ratings remain after matching products.csv.")

    return rating_data.reset_index(drop=True), product_data


ratings, products = _load_data()


# ============================================================
# Encode IDs and build sparse matrices
# ============================================================

product_ids = products["product_id"].to_numpy()
product_to_index = {product_id: index for index, product_id in enumerate(product_ids)}

user_codes, user_ids = pd.factorize(ratings["user_id"], sort=True)
product_codes = ratings["product_id"].map(product_to_index).to_numpy(dtype=np.int32)
rating_values = ratings["rating"].to_numpy(dtype=np.float32)

user_ids = np.asarray(user_ids, dtype=object)
user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}

matrix_shape = (len(user_ids), len(product_ids))

# Explicit matrix stores 1-5 ratings; binary matrix stores purchases/interactions.
user_item_ratings = csr_matrix(
    (rating_values, (user_codes, product_codes)),
    shape=matrix_shape,
    dtype=np.float32,
)

user_item_binary = csr_matrix(
    (np.ones(len(ratings), dtype=np.float32), (user_codes, product_codes)),
    shape=matrix_shape,
    dtype=np.float32,
)

item_user_binary = user_item_binary.transpose().tocsr()
item_interaction_count = np.asarray(user_item_binary.sum(axis=0)).ravel()
global_mean_rating = float(ratings["rating"].mean())


def _display_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in DISPLAY_COLUMNS if column in frame.columns]


def _item_collaborative_scores(item_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return cosine co-purchase scores and shared-user counts for one item."""

    user_indices = item_user_binary.getrow(item_index).indices
    if len(user_indices) == 0:
        return np.zeros(len(products)), np.zeros(len(products), dtype=np.int32)

    shared_counts = np.asarray(
        user_item_binary[user_indices].sum(axis=0)
    ).ravel()

    denominator = np.sqrt(
        item_interaction_count[item_index] * item_interaction_count
    )
    scores = np.divide(
        shared_counts,
        denominator,
        out=np.zeros_like(shared_counts, dtype=np.float64),
        where=denominator > 0,
    )

    scores[item_index] = 0.0
    shared_counts[item_index] = 0.0
    return scores, shared_counts.astype(np.int32)


def _popular_fallback(excluded: set[int], number: int) -> pd.DataFrame:
    """Return well-rated popular products for cold-start situations."""

    candidates = products.copy()
    candidates["_index"] = candidates.index
    candidates = candidates[~candidates["_index"].isin(excluded)].copy()
    candidates["collaborative_score"] = (
        candidates["rating"].fillna(global_mean_rating)
        * np.log1p(candidates["review_count"].fillna(0))
    )
    candidates = candidates.sort_values(
        ["collaborative_score", "rating", "review_count"],
        ascending=[False, False, False],
    ).head(number)

    result = candidates[_display_columns(candidates)].copy()
    result["collaborative_score"] = candidates["collaborative_score"].round(4).to_numpy()
    result["recommendation_reason"] = "Popular and highly rated product"
    return result.reset_index(drop=True)


def users_also_purchased(product_id: str, number: int = 5) -> pd.DataFrame:
    """Find products purchased/rated by users who interacted with an item.

    The source dataset has no order/basket ID. Therefore this function means
    'Users Also Purchased', not products bought in the same transaction.
    """

    product_id = str(product_id).strip()
    if product_id not in product_to_index:
        raise ValueError(f"Unknown product_id: {product_id}")

    number = max(1, min(int(number), 20))
    item_index = product_to_index[product_id]
    scores, shared_counts = _item_collaborative_scores(item_index)

    candidate_indices = np.flatnonzero(shared_counts > 0)
    if len(candidate_indices) == 0:
        return _popular_fallback({item_index}, number)

    candidates = products.iloc[candidate_indices].copy()
    candidates["shared_user_count"] = shared_counts[candidate_indices]
    candidates["collaborative_score"] = scores[candidate_indices]
    candidates = candidates.sort_values(
        ["shared_user_count", "collaborative_score", "rating", "review_count"],
        ascending=[False, False, False, False],
    ).head(number)

    result = candidates[_display_columns(candidates)].copy()
    result["shared_user_count"] = candidates["shared_user_count"].astype(int).to_numpy()
    result["collaborative_score"] = candidates["collaborative_score"].round(4).to_numpy()
    result["recommendation_reason"] = result["shared_user_count"].map(
        lambda count: f"Purchased by {count} shared user{'s' if count != 1 else ''}"
    )
    return result.reset_index(drop=True)


def get_user_history(user_id: str) -> pd.DataFrame:
    """Return products previously rated by one user."""

    user_id = str(user_id).strip()
    if user_id not in user_to_index:
        return pd.DataFrame(columns=[*_display_columns(products), "user_rating"])

    user_index = user_to_index[user_id]
    row = user_item_ratings.getrow(user_index)
    history = products.iloc[row.indices][_display_columns(products)].copy()
    history["user_rating"] = row.data
    return history.sort_values("user_rating", ascending=False).reset_index(drop=True)


def recommend_for_user(user_id: str, number: int = 5) -> pd.DataFrame:
    """Generate personalised item-based collaborative recommendations."""

    user_id = str(user_id).strip()
    number = max(1, min(int(number), 20))

    if user_id not in user_to_index:
        return _popular_fallback(set(), number)

    user_index = user_to_index[user_id]
    history_row = user_item_ratings.getrow(user_index)
    seen_indices = set(history_row.indices.tolist())

    accumulated_scores = np.zeros(len(products), dtype=np.float64)
    evidence_count = np.zeros(len(products), dtype=np.int32)

    for item_index, user_rating in zip(history_row.indices, history_row.data):
        neighbour_scores, shared_counts = _item_collaborative_scores(int(item_index))
        preference_weight = max(float(user_rating) / 5.0, 0.2)
        positive = neighbour_scores > 0
        accumulated_scores[positive] += neighbour_scores[positive] * preference_weight
        evidence_count[positive] += (shared_counts[positive] > 0).astype(np.int32)

    if seen_indices:
        seen_array = np.fromiter(seen_indices, dtype=np.int32)
        accumulated_scores[seen_array] = 0.0

    candidate_indices = np.flatnonzero(accumulated_scores > 0)
    if len(candidate_indices) == 0:
        return _popular_fallback(seen_indices, number)

    candidates = products.iloc[candidate_indices].copy()
    candidates["collaborative_score"] = accumulated_scores[candidate_indices]
    candidates["supporting_history_items"] = evidence_count[candidate_indices]
    candidates = candidates.sort_values(
        ["collaborative_score", "supporting_history_items", "rating", "review_count"],
        ascending=[False, False, False, False],
    ).head(number)

    result = candidates[_display_columns(candidates)].copy()
    result["supporting_history_items"] = (
        candidates["supporting_history_items"].astype(int).to_numpy()
    )
    result["collaborative_score"] = candidates["collaborative_score"].round(4).to_numpy()
    result["recommendation_reason"] = result["supporting_history_items"].map(
        lambda count: f"Based on {count} item{'s' if count != 1 else ''} in your history"
    )
    return result.reset_index(drop=True)


def predict_rating(user_id: str, product_id: str) -> float:
    """Predict a 1-5 rating using the user's collaboratively similar items."""

    user_id = str(user_id).strip()
    product_id = str(product_id).strip()

    if product_id not in product_to_index:
        return round(global_mean_rating, 3)

    item_index = product_to_index[product_id]
    product_mean = float(products.iloc[item_index]["rating"])

    if user_id not in user_to_index:
        return round(float(np.clip(product_mean, 1, 5)), 3)

    user_index = user_to_index[user_id]
    history = user_item_ratings.getrow(user_index)
    if history.nnz == 0:
        return round(float(np.clip(product_mean, 1, 5)), 3)

    similarities, _ = _item_collaborative_scores(item_index)
    weights = similarities[history.indices]
    positive = weights > 0

    if not positive.any():
        return round(float(np.clip(product_mean, 1, 5)), 3)

    prediction = np.average(history.data[positive], weights=weights[positive])
    return round(float(np.clip(prediction, 1, 5)), 3)


# Compatibility names for later hybrid.py and app.py.
def recommend(product_id: str, number: int = 5) -> pd.DataFrame:
    return users_also_purchased(product_id, number)


def collaborative_recommend(user_id: str, number: int = 5) -> pd.DataFrame:
    return recommend_for_user(user_id, number)


if __name__ == "__main__":
    print(f"Users loaded: {len(user_ids):,}")
    print(f"Products loaded: {len(products):,}")
    print(f"Ratings loaded: {len(ratings):,}")
    print(f"Sparse user-item matrix: {user_item_ratings.shape}")

    example_product = products.iloc[0]["product_id"]
    print(f"\nUsers also purchased for {example_product}")
    print(users_also_purchased(example_product, number=3).to_string(index=False))

    active_user_counts = np.diff(user_item_binary.indptr)
    active_user_index = int(np.argmax(active_user_counts))
    example_user = str(user_ids[active_user_index])

    print(f"\nHistory for {example_user}")
    print(get_user_history(example_user).to_string(index=False))

    print(f"\nPersonalised recommendations for {example_user}")
    print(recommend_for_user(example_user, number=3).to_string(index=False))
