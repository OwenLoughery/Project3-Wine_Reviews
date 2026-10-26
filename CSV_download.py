import os 
import sqlite3
import kagglehub
import numpy as np
import pandas as pd

DB_PATH = "wine_reviews.db" #DB pathing 
SCHEMA_PATH = "SCHEMA_SQL.sql"
CSV_130K = "winemag-data-130k-v2.csv" # route for the 130k
CSV_150K = "winemag-data_first150k.csv" # route for the 150k 

COLUMNS = ["country", "description", "designation", "points", "price",
           "province", "region_1", "region_2", "variety", "winery",
           "title", "taster_name", "taster_twitter_handle"]

DEDUPE_KEY = ["description", "winery"]

def download_data():
    path = kagglehub.dataset_download("zynicide/wine-reviews")
    print("Path to dataset files:", path)
    return path

def load_csv(folder, name, source):
    df = pd.read_csv(os.path.join(folder, name), index_col=0)
    df = df.reindex(columns=COLUMNS) # ensures missing have NAN
    df["source"] = source # this will help us keep track of which file its from
    print(f"{source}: loaded {len(df):,} rows")
    return df

def clean(df):
    df = df.copy()
    source = df["source"].iloc[0]

    #handling objects
    text_cols = df.select_dtypes(include=["object", "str"]).columns # grabs object columns
    df[text_cols] = df[text_cols].apply(lambda s: s.str.strip())
    df[text_cols] = df[text_cols].replace("", np.nan) # ensuring each empty text columns is filled with nan

    df = df.dropna(subset=["description", "points"]) # Rows without review text or a score aren't usable

    df = df[df["points"].between(80, 100)].copy()
    df.loc[df["price"] <= 0, "price"] = np.nan

    # Duplicates inside this one file
    before = len(df)
    df = df.drop_duplicates() # drops any duplicates in general
    df = df.drop_duplicates(subset=DEDUPE_KEY) # drops duplicates in terms of their description or winery
    print(f"{source}: removed {before - len(df):,} duplicates, {len(df):,} rows left")
    return df

def merge_datasets(df_130k, df_150k):
    combined = pd.concat([df_130k, df_150k], ignore_index=True) # just combines both of them
    before = len(combined)
    combined = combined.drop_duplicates(subset=DEDUPE_KEY, keep="first") # drops any dublicate s
    combined.index = range(1, len(combined) + 1) 
    print(f"Merged: removed {before - len(combined):,} cross-file duplicates, "
          f"{len(combined):,} unique reviews")
    print(combined["source"].value_counts().to_string())
    return combined

# ensures that our data are building unique tables or unique ids for each and every feature that needs one
# example: wineries.... variety.... etc. 
def build_lookup(df, cols, id_name):
    lookup = df[cols].drop_duplicates().dropna(how="all").reset_index(drop=True)
    lookup.insert(0, id_name, lookup.index + 1)
    return lookup




def parse(df):
    """This """

    # parses the tasters names and cleans any nulls of tasters. Essentially this gives us a list or table of unique tasters 
    tasters = (df[["taster_name", "taster_twitter_handle"]]
               .dropna(subset=["taster_name"])
               .drop_duplicates("taster_name")
               .reset_index(drop=True)
               .rename(columns={"taster_twitter_handle": "twitter_handle"}))
    tasters.insert(0, "taster_id", tasters.index + 1)

    # building unique lists / tables for each of these 
    wineries = build_lookup(df, ["winery"], "winery_id")
    varieties = build_lookup(df, ["variety"], "variety_id")
    locations = build_lookup(df, ["country", "province", "region_1", "region_2"], "location_id")

    # This is attatching our ids to the reviews allowing us to be able to merge them when needed again
    reviews = (df.reset_index(names="review_id")
               .merge(tasters[["taster_id", "taster_name"]], on="taster_name", how="left")
               .merge(wineries, on="winery", how="left")
               .merge(varieties, on="variety", how="left")
               .merge(locations, on=["country", "province", "region_1", "region_2"], how="left"))

    reviews = reviews[["review_id", "title", "designation", "description", "points", "price",
                       "source", "taster_id", "winery_id", "variety_id", "location_id"]]

    for col in ["taster_id", "winery_id", "variety_id", "location_id"]:
        reviews[col] = reviews[col].astype("Int64")

    return {
        "tasters": tasters,
        "wineries": wineries,
        "varieties": varieties,
        "locations": locations,
        "reviews": reviews
    }


def build_database(tables):
    """Builds our data base table"""
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())

    for name, table in tables.items():
        table.to_sql(name, conn, if_exists="append", index=False)
        print(f"Loaded {len(table):>7,} rows into {name}")

    conn.commit()
    return conn


def check(conn):
    """confirming our creation"""
    print(pd.read_sql_query("""
        SELECT l.country, ROUND(AVG(r.points), 2) AS avg_points, COUNT(*) AS n
        FROM reviews r JOIN locations l ON r.location_id = l.location_id
        GROUP BY l.country HAVING n >= 100
        ORDER BY avg_points DESC LIMIT 10;
    """, conn))


if __name__ == "__main__":
    folder = download_data()
    df_130k = clean(load_csv(folder, CSV_130K, "130k"))
    df_150k = clean(load_csv(folder, CSV_150K, "150k"))
    df = merge_datasets(df_130k, df_150k)
    tables = parse(df)
    conn = build_database(tables)
    check(conn)
    conn.close()