from implicit.als import AlternatingLeastSquares
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
import pickle

def train_model(matrix: csr_matrix, factors = 100, iterations = 20,
                regularization = 0.1, alpha = 40):
    als = AlternatingLeastSquares(
        factors=factors,  # size of latent factors
        iterations=iterations,  # number of ALS steps
        regularization=regularization,  # prevents overfitting
        alpha=alpha,  # confidence scaling for implicit feedback
    )

    als.fit(matrix.T.tocsr())

    return als

def recommend(model: AlternatingLeastSquares, matrix: csr_matrix, viewer_id, le_visitors: LabelEncoder,
              le_items:LabelEncoder, n = 10):

    # Sprawdzenie, czy użytkownik istnieje w koderze
    # .classes_ daje tablice kluczy (id) które są unikalne i posortowane
    if viewer_id not in le_visitors.classes_:
        return []

    user_idx = le_visitors.transform([viewer_id])[0]

    item_indices, scores = model.recommend(
        userid= user_idx,
        user_items= matrix[user_idx],
        N=n,
        filter_already_liked_items=True
    )

    recommendations = le_items.inverse_transform(item_indices)

    return list(zip(recommendations, scores))



