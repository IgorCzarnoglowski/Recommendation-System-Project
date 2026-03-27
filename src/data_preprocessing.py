from scipy.sparse import csr_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clear_abnormal_activity(events: pd.DataFrame):
    min_user_interactions = 3
    min_item_interactions = 5
    # Itereujemy dwa razy, bo jak usuniemy np. raz jakieś produkty
    # To wtedy również zmniejszy się ilość klientów z interakcjami
    for _ in range(2):
        item_counts = events['itemid'].value_counts()
        visitor_counts = events['visitorid'].value_counts()

        item_mask = item_counts[item_counts >= min_item_interactions].index
        visitor_mask = visitor_counts[visitor_counts >= min_user_interactions].index

        events = events[events['itemid'].isin(item_mask)]
        events = events[events['visitorid'].isin(visitor_mask)]

    return events

def create_matrix(events: pd.DataFrame):
    # Laber encoder oddzielnie, aby nie tworzyć zbyt dużej macierzy
    le_viewers = LabelEncoder()
    le_items = LabelEncoder()

    events['viewerid'] = le_viewers.fit_transform(events['viewerid'])
    events['itemid'] = le_items.fit_transform(events['itemid'])

    




if __name__ == "__main__":
    from data_loader import load_events
    print('---Przed filtrem:')
    print(load_events()['event'].value_counts())
    print('---Po filtrze:')
    print(clear_abnormal_activity(load_events())['event'].value_counts())
