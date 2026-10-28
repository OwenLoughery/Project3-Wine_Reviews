import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# built by dataset_cleaning.ipynb (same review_id as wine_reviews.db)
INPUT_PATH = "data/wine_clean.parquet"
STYLE_GROUPS_PATH = "wine_style_groups_final.csv"
OUTPUT_PATH = "data/processed_reviews.parquet"

# only hits the network the first time, when the nltk data isn't on this machine yet
for pkg, path in [("punkt_tab", "tokenizers/punkt_tab"), ("stopwords", "corpora/stopwords"),
                  ("wordnet", "corpora/wordnet.zip")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

LEMMATIZER = WordNetLemmatizer()

# common words that are used and commonly shared between each group that we found in our data analysis notebook
WINE_WORDS = {"wine", "flavor", "aroma", "palate", "finish", "drink", "note", "show", "acidity"}

# these are words that are within the style names that we want to avoid getting rid of
# sparkling / champagne / port describe the wine itself, so they stay in the reviews
KEEP = {"red", "white", "apple", "blend", "style", "other", "de", "di", "del", "la",
        "black", "orange", "melon", "vital", "grain", "diamond", "mission", "du",
        "sparkling", "champagne", "port"}

# short names for grapes that aren't part of any variety name (zin = 31% of Zinfandel reviews, cab = 14% of Cabernet)
# varietyname is the placeholder dataset_cleaning puts where it masked a grape name
NICKNAMES = {"cab", "zin", "chard", "sauv", "gewürz", "gewurz", "grüner", "gruner", "varietyname"}

def load_reviews(path = INPUT_PATH):
    """
    loads the cleaned dataset from dataset_cleaning.ipynb (already has grape, style_group, split, etc.)
    """
    return pd.read_parquet(path)

def load_style_map(path=STYLE_GROUPS_PATH):
    """
    Will return a map of each style and which varieties belong in each style_group type
    """
    groups = pd.read_csv(path)
    groups["variety"] = groups["varieties"].str.split("; ")
    style_map = groups.explode("variety")[["variety", "style_group", "batch"]] # explodes distributes the list into variety and creates new columns
    style_map["variety"] = style_map["variety"].str.strip()
    return style_map.reset_index(drop=True)

# ------- Applying the Text Cleaning -------
# Essentially all of these are fixing up all of the tokenization and descriptions similar to how we did it in data analysis
def base_tokens(text):
    return [LEMMATIZER.lemmatize(t) for t in word_tokenize(text.lower()) if t.isalpha()]

def name_words(names):
    """Turn variety names into the same token form as the reviews."""
    return {t for name in names for t in base_tokens(name)}

def build_stopwords(style_map):
    """English stopwords + wine filler + every variety-name word (minus KEEP) + grape nicknames."""
    english = set(stopwords.words("english"))
    variety_words = name_words(style_map["variety"]) - KEEP
    return english | WINE_WORDS | variety_words | NICKNAMES


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
    df = load_reviews()
    style_map = load_style_map()

    english = set(stopwords.words("english"))
    all_stops = build_stopwords(style_map)
    print(f"Stopwords: {len(all_stops):,} total")

    print("Tokenizing (takes a few minutes)...")
    # tokens_raw is the original text (to show the leakage before), tokens is the masked text with every stop word removed
    df["tokens_raw"] = df["description"].apply(base_tokens).apply(remove_stops, stops=english)
    df["tokens"] = df["description_masked"].apply(base_tokens).apply(remove_stops, stops=all_stops)
    df["clean_text"] = df["tokens"].str.join(" ")
    df["clean_designation"] = (df["designation"].fillna("")
                                 .apply(base_tokens)
                                 .apply(remove_stops, stops=all_stops)
                                 .str.join(" "))

    report = pd.concat([leakage_report(df, style_map, "tokens_raw"),
                        leakage_report(df, style_map, "tokens")], axis=1)
    print(report.sort_values("tokens_raw", ascending=False).to_string())

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")
    return df # this will return the df and create a parquet file that is already setup


if __name__ == "__main__":
    process()
