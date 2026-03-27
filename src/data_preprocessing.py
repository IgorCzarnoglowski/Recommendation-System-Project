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
    le_visitors = LabelEncoder()
    le_items = LabelEncoder()

    events['visitorid'] = le_visitors.fit_transform(events['visitorid'])
    events['itemid'] = le_items.fit_transform(events['itemid'])

    # używam nunique, bo daje mi to unikalne wartości zamiast ilości wierszy
    # bezpieczniejsza wersja
    matrix = csr_matrix((events['weight'], (events['visitorid'], events['itemid'])),
                        shape=(events['visitorid'].nunique(), events['itemid'].nunique()))

    # manualna wersja encodowania i robienia macierzy
    '''
    user_ids = interactions_cf["visitorid"].unique()
    item_ids = interactions_cf["itemid"].unique()

    user_to_idx = {u:i for i,u in enumerate(user_ids)}
    item_to_idx = {it:i for i,it in enumerate(item_ids)}
    
    rows = interactions_cf["visitorid"].map(user_to_idx).values
    cols = interactions_cf["itemid"].map(item_to_idx).values
    data = interactions_cf["implicit_score"].values.astype(np.float32)
    
    UI = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(item_ids)))
    '''

    return matrix






if __name__ == "__main__":
    from data_loader import load_events
    print('---Przed filtrem:')
    print(load_events()['event'].value_counts())
    print('---Po filtrze:')
    after_filter = clear_abnormal_activity(load_events())
    print(after_filter['event'].value_counts())
    print('---Wielkość macierzy')
    print(create_matrix(after_filter).shape)
    

