from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, recall_score
from sklearn.metrics.pairwise import linear_kernel

PROJECT_DIR = Path(__file__).resolve().parents[1]
RATINGS_PATH = PROJECT_DIR / "data" / "ratings.csv"
PRODUCTS_PATH = PROJECT_DIR / "data" / "products.csv"
RESULTS_PATH = PROJECT_DIR / "evaluation" / "evaluation_results.csv"
PREDICTIONS_PATH = PROJECT_DIR / "evaluation" / "evaluation_predictions.csv"

RANDOM_STATE = 42
MAX_EVALUATION_USERS = 200
RELEVANCE_THRESHOLD = 4.0
MODELS = ["Content-Based", "Collaborative", "Hybrid"]
METRICS = ["RMSE", "MAE", "Precision", "Recall", "F1-score"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(RATINGS_PATH, usecols=["user_id", "product_id", "rating"])
    products = pd.read_csv(PRODUCTS_PATH)

    ratings["user_id"] = ratings["user_id"].astype(str).str.strip()
    ratings["product_id"] = ratings["product_id"].astype(str).str.strip()
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    ratings = ratings.dropna(subset=["user_id", "product_id", "rating"])
    ratings = ratings[ratings["rating"].between(1, 5)].copy()

    if ratings.duplicated(["user_id", "product_id"]).any():
        ratings = ratings.groupby(["user_id", "product_id"], as_index=False).agg(
            rating=("rating", "mean")
        )

    products["product_id"] = products["product_id"].astype(str).str.strip()
    products = products.drop_duplicates("product_id").reset_index(drop=True)
    ratings = ratings[ratings["product_id"].isin(set(products["product_id"]))].reset_index(drop=True)
    return ratings, products


def leave_one_out_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    interaction_count = ratings.groupby("user_id")["product_id"].size()
    active_users = interaction_count[interaction_count >= 2].index.to_numpy()
    if len(active_users) == 0:
        raise ValueError("Evaluation requires users with at least two ratings.")

    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(MAX_EVALUATION_USERS, len(active_users))
    selected_users = rng.choice(active_users, size=sample_size, replace=False)
    selected_ratings = ratings[ratings["user_id"].isin(selected_users)]
    test = selected_ratings.groupby("user_id", group_keys=False).sample(
        n=1, random_state=RANDOM_STATE
    ).copy()
    train = ratings.drop(index=test.index).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


class OfflineCollaborativeModel:
    """Item-based collaborative model trained only on the training split."""

    def __init__(self, train: pd.DataFrame, all_user_ids: np.ndarray, all_product_ids: np.ndarray):
        self.user_ids = np.asarray(all_user_ids, dtype=object)
        self.product_ids = np.asarray(all_product_ids, dtype=object)
        self.user_to_index = {u: i for i, u in enumerate(self.user_ids)}
        self.product_to_index = {p: i for i, p in enumerate(self.product_ids)}
        rows = pd.Categorical(train["user_id"], categories=self.user_ids).codes.astype(np.int32)
        cols = pd.Categorical(train["product_id"], categories=self.product_ids).codes.astype(np.int32)
        vals = train["rating"].to_numpy(dtype=np.float32)
        shape = (len(self.user_ids), len(self.product_ids))
        self.user_item_ratings = csr_matrix((vals, (rows, cols)), shape=shape, dtype=np.float32)
        self.user_item_binary = csr_matrix((np.ones(len(train), dtype=np.float32), (rows, cols)), shape=shape, dtype=np.float32)
        self.item_user_binary = self.user_item_binary.transpose().tocsr()
        self.item_counts = np.asarray(self.user_item_binary.sum(axis=0)).ravel()
        self.global_mean = float(train["rating"].mean())
        self.product_means = train.groupby("product_id")["rating"].mean().to_dict()

    @staticmethod
    def _intersection_count(left: np.ndarray, right: np.ndarray) -> int:
        i = j = common = 0
        while i < len(left) and j < len(right):
            if left[i] == right[j]:
                common += 1; i += 1; j += 1
            elif left[i] < right[j]:
                i += 1
            else:
                j += 1
        return common

    def predict(self, user_id: str, product_id: str) -> float:
        fallback = float(self.product_means.get(product_id, self.global_mean))
        if user_id not in self.user_to_index or product_id not in self.product_to_index:
            return float(np.clip(fallback, 1, 5))
        user_index = self.user_to_index[user_id]
        target_index = self.product_to_index[product_id]
        history = self.user_item_ratings.getrow(user_index)
        if history.nnz == 0 or self.item_counts[target_index] == 0:
            return float(np.clip(fallback, 1, 5))
        target_users = self.item_user_binary.getrow(target_index).indices
        similarities, neighbour_ratings = [], []
        for history_index, history_rating in zip(history.indices, history.data):
            history_users = self.item_user_binary.getrow(int(history_index)).indices
            shared = self._intersection_count(target_users, history_users)
            if shared == 0:
                continue
            denominator = np.sqrt(self.item_counts[target_index] * self.item_counts[history_index])
            if denominator <= 0:
                continue
            similarities.append(shared / denominator)
            neighbour_ratings.append(float(history_rating))
        if not similarities:
            return float(np.clip(fallback, 1, 5))
        prediction = np.average(neighbour_ratings, weights=similarities)
        evidence = min(sum(similarities), 1.0)
        prediction = evidence * prediction + (1 - evidence) * fallback
        return float(np.clip(prediction, 1, 5))


class OfflineContentModel:
    """Fast content-based model using product metadata similarity."""

    def __init__(self, train: pd.DataFrame, products: pd.DataFrame, selected_users: set[str] | None = None):
        self.products = products.reset_index(drop=True).copy()
        self.product_to_index = dict(zip(self.products["product_id"].astype(str), self.products.index))
        self.global_mean = float(train["rating"].mean())
        self.product_means = train.groupby("product_id")["rating"].mean().to_dict()
        history_source = train if selected_users is None else train[train["user_id"].astype(str).isin(selected_users)]
        self.user_history = {str(user): group.copy() for user, group in history_source.groupby("user_id")}
        self.product_rows = {str(row.product_id): row for row in self.products.itertuples(index=False)}

    @staticmethod
    def similarity(target: pd.Series, history_item: pd.Series) -> float:
        score = 0.0
        if str(target.get("subcategory", "")).casefold() == str(getattr(history_item, "subcategory", "")).casefold():
            score += 0.60
        if str(target.get("category", "")).casefold() == str(getattr(history_item, "category", "")).casefold():
            score += 0.25
        if str(target.get("brand", "")).casefold() == str(getattr(history_item, "brand", "")).casefold():
            score += 0.15
        return score

    def predict(self, user_id: str, product_id: str) -> float:
        fallback = float(self.product_means.get(product_id, self.global_mean))
        if user_id not in self.user_history or product_id not in self.product_to_index:
            return float(np.clip(fallback, 1, 5))
        target = self.products.iloc[self.product_to_index[product_id]]
        history = self.user_history[user_id]
        weights, values = [], []
        for item in history.itertuples(index=False):
            item_id = str(item.product_id)
            history_item = self.product_rows.get(item_id)
            if history_item is None:
                continue
            similarity = self.similarity(target, history_item)
            if similarity > 0:
                weights.append(similarity * max(float(item.rating) / 5.0, 0.2))
                values.append(float(item.rating))
        if not weights:
            return float(np.clip(fallback, 1, 5))
        return float(np.clip(np.average(values, weights=weights), 1, 5))


def evaluate_metrics(actual: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    actual_relevant = actual >= RELEVANCE_THRESHOLD
    predicted_relevant = predictions >= RELEVANCE_THRESHOLD
    return {
        "RMSE": float(np.sqrt(np.mean((actual - predictions) ** 2))),
        "MAE": float(mean_absolute_error(actual, predictions)),
        "Precision": float(precision_score(actual_relevant, predicted_relevant, zero_division=0)),
        "Recall": float(recall_score(actual_relevant, predicted_relevant, zero_division=0)),
        "F1-score": float(f1_score(actual_relevant, predicted_relevant, zero_division=0)),
    }


def main() -> None:
    print("Loading evaluation data...")
    ratings, products = load_data()
    train, test = leave_one_out_split(ratings)

    all_user_ids = ratings["user_id"].drop_duplicates().sort_values().to_numpy()
    all_product_ids = products["product_id"].to_numpy()
    collaborative = OfflineCollaborativeModel(train, all_user_ids, all_product_ids)
    selected_users = set(test["user_id"].astype(str))
    content = OfflineContentModel(train, products, selected_users)

    predictions = {model: np.empty(len(test), dtype=float) for model in MODELS}
    for position, row in enumerate(test.itertuples(index=False)):
        user_id, product_id = str(row.user_id), str(row.product_id)
        content_pred = content.predict(user_id, product_id)
        collab_pred = collaborative.predict(user_id, product_id)
        product_mean = float(pd.to_numeric(products.loc[products["product_id"].eq(product_id), "rating"], errors="coerce").iloc[0]) if product_id in set(products["product_id"]) else float(train["rating"].mean())
        hybrid_pred = 0.35 * content_pred + 0.55 * collab_pred + 0.10 * product_mean
        predictions["Content-Based"][position] = content_pred
        predictions["Collaborative"][position] = collab_pred
        predictions["Hybrid"][position] = float(np.clip(hybrid_pred, 1, 5))

    actual = test["rating"].to_numpy(dtype=float)
    rows = []
    for model in MODELS:
        metrics = evaluate_metrics(actual, predictions[model])
        for metric in METRICS:
            rows.append({"model": model, "metric": metric, "value": metrics[metric]})

    results = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)

    details = test.copy()
    for model in MODELS:
        details[f"{model.lower().replace('-', '_').replace(' ', '_')}_prediction"] = np.round(predictions[model], 4)
    details.to_csv(PREDICTIONS_PATH, index=False)

    print("\nModel comparison")
    print("=" * 60)
    print(results.pivot(index="metric", columns="model", values="value").to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nEvaluation users: {len(test):,}")
    print(f"Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
