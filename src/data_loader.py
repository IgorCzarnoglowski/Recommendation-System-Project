import pandas as pd
import os

curr_path = os.path.dirname(__file__)

def load_events():
    # Sposób na znalezienie pliku w innym folderze niż plik .py
    df = pd.read_csv(os.path.relpath('..\\data\\events.csv', curr_path))

    # Ponieważ kolumna timestamp jest w UNIX timestamp (milisekundach od 1 stycznia, 1970 roku)
    #Trzeba to zmienić na naszą date
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.rename(columns = {'timestamp': 'time'})

    #mapujemy wagi do odpowiednich eventów
    weight_map = {'view': 1, 'addtocart': 3, 'transaction': 5}
    df['weight'] = df['event'].map(weight_map)

    df = df.dropna(subset=['visitorid', 'itemid'])
    df["visitorid"] = df["visitorid"].astype(int)
    df["itemid"] = df["itemid"].astype(int)

    return df


def load_item_properties():
    part1 = pd.read_csv(os.path.relpath('..\\data\\item_properties_part1.csv', curr_path))
    part2 = pd.read_csv(os.path.relpath('..\\data\\item_properties_part2.csv', curr_path))

    df = pd.concat([part1, part2], ignore_index = True)
    df["itemid"] = df["itemid"].astype(int)
    #Bierzemy tylko pod uwagę kolumnę property, bo do ALS tylko ona potencjalnie jest przydatna
    readable = df[df["property"].isin(["categoryid", "available"])]

    # Pivotujmey aby każda właściwość staje się kolumną
    pivoted = (readable
               .drop_duplicates(subset=["itemid", "property"], keep="last")
               .pivot(index="itemid", columns="property", values="value")
               .reset_index())

    pivoted.columns.name = None

    # Typy danych
    if "available" in pivoted.columns:
        pivoted["available"] = pd.to_numeric(pivoted["available"], errors="coerce").fillna(0).astype(int)

    if "categoryid" in pivoted.columns:
        pivoted["categoryid"] = pd.to_numeric(pivoted["categoryid"], errors="coerce")

    return pivoted

