from dataclasses import dataclass
from implicit.als import AlternatingLeastSquares
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

@dataclass
class RecommenderModel:
    model: AlternatingLeastSquares
    le_viewers: LabelEncoder
    le_items: LabelEncoder

def train_model(matrix: csr_matrix, factors = 100, iterations = 20,
                regularization = 0.1, alpha = 40) -> AlternatingLeastSquares:
    als = AlternatingLeastSquares(
        factors=factors,  # size of latent factors
        iterations=iterations,  # number of ALS steps
        regularization=regularization,  # prevents overfitting
        alpha=alpha,  # confidence scaling for implicit feedback
    )

    als.fit(matrix)

    return als

def recommend(model: RecommenderModel, matrix: csr_matrix, viewer_id, n = 10) -> list:


    # Sprawdzenie, czy użytkownik istnieje w koderze
    # .classes_ daje tablice kluczy (id) które są unikalne i posortowane
    if viewer_id not in model.le_viewers.classes_:
        return []

    user_idx = model.le_viewers.transform([viewer_id])[0]

    item_indices, scores = model.model.recommend(
        userid= user_idx,
        user_items= matrix[user_idx],
        N=n,
        filter_already_liked_items=True
    )

    recommendations = model.le_items.inverse_transform(item_indices)


    return list(zip(recommendations, scores))

def similar_items(rec: RecommenderModel,
                  product_id: int,
                  n: int = 10) -> list[tuple]:

    if product_id not in rec.le_items.classes_:
        return []

    pid = rec.le_items.transform([product_id])[0]

    similar_indices, scores = rec.model.similar_items(pid, N=n + 1)

    product_ids = rec.le_items.inverse_transform(similar_indices)

    return [
        (pid_result, score)
        for pid_result, score in zip(product_ids, scores)
        if pid_result != product_id
    ]


def save(rec: RecommenderModel, name: str = "als_model") -> None:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(rec, f)


def load(name: str = "als_model") -> RecommenderModel:
    path = MODELS_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)




