import os 
import sqlite3
import kagglehub
import pandas as pd

DB_PATH = "wine_reviews.db"
SCHEMA_PATH = "SCHEMA_SQL.sql"
CSV_NAME = "winemag-data_first150k.csv"

def download_data():
    path = kagglehub.dataset_download("zynicide/wine-reviews")
    print("Path to dataset files:", path)
    csv_path = os.path.join(path, CSV_NAME)
    df = pd.read_csv(csv_path, index_col=0)
    print("Columns:", df.columns.tolist())
    df = df.drop_duplicates()
    print("Rows after dedupe:", len(df))
    return df

def build_lookup(df, cols, id_name):
    """Unique combos of `cols`, numbered 1..n as `id_name`."""
    lookup = df[cols].drop_duplicates().dropna(how="all").reset_index(drop=True)
    lookup.insert(0, id_name, lookup.index + 1)
    return lookup

def parse(df):
    wineries = build_lookup(df, ["winery"], "winery_id")
    varieties = build_lookup(df, ["variety"], "variety_id")
    locations = build_lookup(df, ["country", "province", "region_1", "region_2"], "location_id")

    reviews = (df.reset_index(names="review_id")
               .merge(wineries, on="winery", how="left")
               .merge(varieties, on="variety", how="left")
               .merge(locations, on=["country", "province", "region_1", "region_2"], how="left"))

    reviews = reviews[["review_id", "designation", "description", "points", "price",
                       "winery_id", "variety_id", "location_id"]]

    for col in ["winery_id", "variety_id", "location_id"]:
        reviews[col] = reviews[col].astype("Int64")

    return {
        "wineries": wineries,
        "varieties": varieties,
        "locations": locations,
        "reviews": reviews
    }


def build_database(tables):
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())

    for name, table in tables.items():
        table.to_sql(name, conn, if_exists="append", index=False)
        print(f"Loaded {len(table):>7,} rows into {name}")

    conn.commit()
    return conn

def check(conn):
    print(pd.read_sql_query("""
        SELECT l.country, ROUND(AVG(r.points), 2) AS avg_points, COUNT(*) AS n
        FROM reviews r JOIN locations l ON r.location_id = l.location_id
        GROUP BY l.country HAVING n >= 100
        ORDER BY avg_points DESC LIMIT 10;
    """, conn))


if __name__ == "__main__":
    df = download_data()
    tables = parse(df)
    conn = build_database(tables)
    check(conn)
    conn.close()