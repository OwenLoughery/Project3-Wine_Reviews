import sqlite3

import nltk
import pandas as pd 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

DB_PATH = "wine_reviews.db"
STYLE_GROUPS_PATH = "wine_style_groups_final.csv"
OUTPUT_PATH = "processed_reviews.pkl"

for pkg in ["punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True) 

LEMMATIZER = WordNetLemmatizer()

# common words that are used and commonly shared between each group that we found in our data analysis notebook
WINE_WORDS = {"wine", "flavor", "aroma", "palate", "finish", "drink", "note", "show", "acidity"} 

# these are words that are within the style names that we want to avoid getting rid of
KEEP = {"red", "white", "apple", "blend", "style", "other", "de", "di", "del", "la",
        "black", "orange", "melon", "vital", "grain", "diamond", "mission", "du"}

def load_reviews(db_path = DB_PATH):
    """
    loads the data base and joins them into one dataframe
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
    SELECT r.review_id, r.source,
        l.country, l.province, l.region_1, l.region_2,
        w.winery, v.variety, r.designation,
        r.points, r.price, r.description
    FROM reviews  as r
    LEFT JOIN wineries as w ON r.winery_id = w.winery_id
    LEFT JOIN varieties as v ON r.variety_id = v.variety_id
    LEFT JOIN locations as l ON r.location_id = l.location_id
    ORDER BY r.review_id
    """, conn)
    conn.close()
    return df

def load_style_map(path=STYLE_GROUPS_PATH):
    """
    Will return a map of each style and which varieties belong in each style_group type
    """
    groups = pd.read_csv(path)
    groups["variety"] = groups["varieties"].str.split("; ")
    style_map = groups.explode("variety")[["variety", "style_group", "batch"]] # explodes distributes the list into variety and creates new columns 
    style_map["variety"] = style_map["variety"].str.strip()
    return style_map.reset_index(drop=True)

def add_style_groups(reviews, style_map):
    "adding the new column: style_group"
    df = reviews.merge(style_map, on="variety", how="inner")
    print(f"Matched {len(df):,} of {len(reviews):,} reviews to a style group "
          f"({len(reviews) - len(df):,} dropped)")
    return df

# ------- Applying the Text Cleaning -------
# Essentially all of these are fixing up all of the tokenization and descriptions similar to how we did it in data analysis
def base_tokens(text):
    return [LEMMATIZER.lemmatize(t) for t in word_tokenize(text.lower()) if t.isalpha()]

def name_words(names):
    """Turn variety names into the same token form as the reviews."""
    return {t for name in names for t in base_tokens(name)}

def build_stopwords(style_map):
    """English stopwords + wine filler + every variety-name word (minus KEEP)."""
    english = set(stopwords.words("english"))
    variety_words = name_words(style_map["variety"]) - KEEP
    return english | WINE_WORDS | variety_words


def remove_stops(tokens, stops):
    return [t for t in tokens if t not in stops]

def leakage_report(df, style_map, col):
    """% of each style group's reviews that mention one of its OWN variety words....."""
    rates = {}
    for style, varieties in style_map.groupby("style_group")["variety"]:
        own = name_words(varieties) - KEEP
        sub = df[df["style_group"] == style]
        rates[style] = round(100 * sub[col].apply(lambda toks: bool(own & set(toks))).mean(), 1)
    return pd.Series(rates, name=col)

def process():
    reviews = load_reviews()
    style_map = load_style_map()
    df = add_style_groups(reviews, style_map)

    english = set(stopwords.words("english"))
    all_stops = build_stopwords(style_map)
    print(f"Stopwords: {len(all_stops):,} total")

    print("Tokenizing (takes a few minutes)...")
    base = df["description"].apply(base_tokens)
    df["tokens_raw"] = base.apply(remove_stops, stops=english)
    df["tokens"] = base.apply(remove_stops, stops=all_stops)
    df["clean_text"] = df["tokens"].str.join(" ")
    df["clean_designation"] = (df["designation"].fillna("")
                                 .apply(base_tokens)
                                 .apply(remove_stops, stops=all_stops)
                                 .str.join(" "))

    report = pd.concat([leakage_report(df, style_map, "tokens_raw"),
                        leakage_report(df, style_map, "tokens")], axis=1)
    print(report.sort_values("tokens_raw", ascending=False).to_string())

    df.to_pickle(OUTPUT_PATH)
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")
    return df # this will return the df and create a pkl / df file that is already setup


if __name__ == "__main__":
    process()