import numpy as np
import pandas as pd
from als_model import RecommenderModel, recommend
from scipy.sparse import csr_matrix

def precision_at_k(recommended: RecommenderModel, matrix:csr_matrix, test: pd.DataFrame, k=10, n=1000) -> float:
    # Wybieramy uzytkowników na podstawie ich akcji
    # Najsilniejszym bodźcem jest 'transaction', więc na tym się skupimy
    actual = (test[test["event_type"] == "transaction"]
              .groupby("visitorid")["itemid"]
              .apply(set)
              .to_dict())

    # Filtrujemy naszych klientów, aby mieli przynajmniej jedną akcje transakcji
    eliglibele_visitors = [
        id for id in actual if id in recommended.le_visitors.classes_ and len(actual[id]) > 0]


    sample_viewers = np.random.choice(eliglibele_visitors, size=min(n, len(eliglibele_visitors)), replace = False)

    precisions = []
    for vid in sample_viewers:
        rec = [id for id, _ in recommend(recommended, matrix, vid, k)]
        # & działa tylko dla setów stąd set(rec)
        hits = len(set(rec) & actual[vid])
        precisions.append(hits/k)

    # Bierzemy średnią dla wszystkich użytkowników p@k
    return float(np.mean(precisions))



