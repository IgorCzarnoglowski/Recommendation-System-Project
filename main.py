from src.data_loader import load_events
from src.data_preprocessing import (create_matrix,
                                clear_abnormal_activity,
                                train_test_split_temporal)
from src.als_model import RecommenderModel, train_model, save, load, recommend
from src.evaluate import evaluate_model


def main():
    print("1. Ładowanie danych...")
    events = load_events()

    print("2. Filtrowanie...")
    events = clear_abnormal_activity(events)

    print("3. Split czasowy...")
    train_events, test_events = train_test_split_temporal(events, test_days=7)

    print("4. Budowanie macierzy...")
    matrix, le_viewers, le_items = create_matrix(train_events)

    print("5. Trening modelu...")
    als = train_model(matrix)
    rec = RecommenderModel(model=als, le_viewers=le_viewers, le_items=le_items)

    # Aby zapisany model użyć, wystarczy odhashować tą część i zahashować trening modelu i zapis modelu
    #print("5. Ładowanie modelu")
    #rec = load('als_model')

    print("6. Ewaluacja...")
    k = 10
    evaluate = evaluate_model(rec, matrix, test_events, k=k, n=1000)

    print("7. Zapis modelu...")
    save(rec)


    print("\nPrzykładowe rekomendacje:")
    sample_user = int(test_events["visitorid"].iloc[0])
    recs = recommend(rec, matrix, sample_user, n=5)
    print(f"User {sample_user}: {recs}")
    print(f"precison@{k}: {evaluate['precision']}\nrecall@{k}: {evaluate['recall']}")


if __name__ == "__main__":
    main()