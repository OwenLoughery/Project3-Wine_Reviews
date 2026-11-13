"""
Runs the grape comparisons and saves the results for grape_difference_results.ipynb to read.

To add a comparison, edit the lists below and run one of these:
    python run_grape_differences.py                    (everything but within_reviewer, about 7 minutes)
    python run_grape_differences.py within_reviewer    (one reviewer, two grapes, about 3 hours)

Comparisons run on all reviews and on only the reviews that name a taster. Reviews with no taster
separate more easily and we can't check why, so we only trust patterns in both.

Most countries are covered by one main reviewer, so a difference between two countries could just be
two writing styles. The same-reviewer tests at the end are what let us tell those apart.
"""

import os
import sys
import time
from itertools import combinations, permutations

import grape_differences as gd
import numpy as np
import pandas as pd

RESULTS = "results"
SCORES_PATH = f"{RESULTS}/grape_difference_scores.csv"
WORDS_PATH = f"{RESULTS}/grape_difference_top_words.csv.gz"
TRANSFER_PATH = f"{RESULTS}/grape_transfer_scores.csv"
REVIEWER_SCORES_PATH = f"{RESULTS}/grape_same_reviewer_scores.csv"
REVIEWER_WORDS_PATH = f"{RESULTS}/grape_same_reviewer_top_words.csv"
WITHIN_SCORES_PATH = f"{RESULTS}/grape_within_reviewer_scores.csv"
WITHIN_WORDS_PATH = f"{RESULTS}/grape_within_reviewer_top_words.csv"

# Chosen before seeing any results: (kind, grape_a, grape_b, place_a, place_b)
# These use every review the smaller side has, so their scores can't be compared to the grid's.
PLANNED = [
    ("red vs white", "Pinot Noir", "Chardonnay", "California", "California"),
    ("different reds", "Pinot Noir", "Cabernet Sauvignon", "California", "California"),
    ("similar reds", "Cabernet Sauvignon", "Merlot", "California", "California"),
    ("different whites", "Chardonnay", "Sauvignon Blanc", "California", "California"),
    ("similar whites", "Pinot Gris", "Pinot Blanc", "Oregon", "Oregon"),
    ("same grape, two regions", "Pinot Noir", "Pinot Noir", "California", "Burgundy"),
]
PLANNED_SEEDS = range(5)

# Every pair of grapes with enough reviews in a place, at one fixed size so scores are comparable.
GRID_PLACES = [
    (None, "all"),
    ("US", "country"),
    ("France", "country"),
    ("Italy", "country"),
    ("Chile", "country"),
    ("South Africa", "country"),
    ("Australia", "country"),
    ("Argentina", "country"),
    ("New Zealand", "country"),
    ("Spain", "country"),
    ("Austria", "country"),
    ("California", "province"),
    ("Washington", "province"),
]
GRID_MIN_REVIEWS = 200
GRID_SEEDS = range(3)
N_PER_SIDE = 200

# Near means two countries on the same continent, far means different continents.
CONTINENT = {
    "US": "North America",
    "Canada": "North America",
    "Chile": "South America",
    "Argentina": "South America",
    "France": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "Portugal": "Europe",
    "Germany": "Europe",
    "Austria": "Europe",
    "Greece": "Europe",
    "Australia": "Oceania",
    "New Zealand": "Oceania",
    "South Africa": "Africa",
    "Israel": "Asia",
}

# A model trained in one country and tested in another: same-color pairs both countries have plenty of.
TRANSFER_MIN_REVIEWS = 300
TRANSFER_SEEDS = range(5)

# The small tests: one reviewer's grape in two countries, and two reviewers on one grape in one place.
REVIEWER_MIN_REVIEWS = 100
REVIEWER_FIXED_SIZE = 80
REVIEWER_SEEDS = range(3)

# The control splits one reviewer's grape in half at random, so each half needs the fixed size.
CONTROL_MIN_REVIEWS = 180
CONTROL_SEED = 4401


def near_or_far(country_a, country_b):
    return "near" if CONTINENT[country_a] == CONTINENT[country_b] else "far"


def run_planned(reviews):
    """The hand-picked comparisons, at full size."""
    rows, words = [], []
    for kind, grape_a, grape_b, place_a, place_b in PLANNED:
        row, pair_words = gd.compare_pair(
            reviews, grape_a, grape_b, place_a, place_b, seeds=PLANNED_SEEDS
        )
        rows.append({"run": "planned", "kind": kind, **row})
        words.append(pair_words.assign(run="planned", kind=kind))
    return pd.DataFrame(rows), pd.concat(words, ignore_index=True)


def run_grids(reviews):
    """Every pair in each of GRID_PLACES, at N_PER_SIDE reviews per grape."""
    color = reviews.drop_duplicates("grape").set_index("grape")["color"]
    scores, words = [], []
    for place, level in GRID_PLACES:
        pairs = gd.region_pairs(
            reviews, place, level, min_reviews=GRID_MIN_REVIEWS, same_color=False
        )
        place_scores, place_words = gd.run_grid(
            reviews, pairs, level, n_per_side=N_PER_SIDE, seeds=GRID_SEEDS
        )
        if not place_scores.empty:
            scores.append(place_scores)
            words.append(place_words)

    def kind_of(grape_a, grape_b):
        return "red vs white" if color[grape_a] != color[grape_b] else f"{color[grape_a]}s"

    scores, words = pd.concat(scores), pd.concat(words)
    for table in [scores, words]:
        table["run"] = "grid"
        table["kind"] = [
            kind_of(a, b) for a, b in zip(table["grape_a"], table["grape_b"], strict=True)
        ]
    return scores, words


def run_near_far(reviews):
    """The same grape in two countries, for every pair of countries with enough reviews of it."""
    counts = reviews.groupby(["grape", "country"]).size()
    rows, words = [], []
    for grape, by_country in counts.groupby(level="grape"):
        countries = [
            country
            for (_, country), n in by_country.items()
            if n >= GRID_MIN_REVIEWS and country in CONTINENT
        ]
        for country_a, country_b in combinations(sorted(countries), 2):
            try:
                row, pair_words = gd.compare_pair(
                    reviews,
                    grape,
                    grape,
                    country_a,
                    country_b,
                    level="country",
                    n_per_side=N_PER_SIDE,
                    seeds=GRID_SEEDS,
                )
            except ValueError:
                continue
            kind = near_or_far(country_a, country_b)
            rows.append({"run": "near_far", "kind": kind, **row})
            words.append(pair_words.assign(run="near_far", kind=kind))
    return pd.DataFrame(rows), pd.concat(words, ignore_index=True)


def run_transfers(reviews):
    """Trains a same-color pair in one country and tests it in another, in both directions."""
    color = reviews.drop_duplicates("grape").set_index("grape")["color"]
    counts = reviews.groupby(["country", "grape"]).size()
    big = counts[counts >= TRANSFER_MIN_REVIEWS].reset_index()
    rows = []
    for (grape_a, grape_b) in combinations(sorted(big["grape"].unique()), 2):
        if color[grape_a] != color[grape_b]:
            continue
        has_a = set(big.loc[big["grape"].eq(grape_a), "country"])
        has_b = set(big.loc[big["grape"].eq(grape_b), "country"])
        for trained_in, tested_in in permutations(sorted(has_a & has_b), 2):
            try:
                row = gd.transfer_pair(
                    reviews, grape_a, grape_b, trained_in, tested_in, seeds=TRANSFER_SEEDS
                )
            except ValueError:
                continue
            rows.append(
                {
                    "kind": near_or_far(trained_in, tested_in),
                    "color": color[grape_a],
                    **row,
                }
            )
    return pd.DataFrame(rows)


def run_small_comparison(side_a, side_b, description):
    """One small test at a fixed size (comparable across tests) and at full size (more reliable)."""
    fixed_row, _ = gd.cross_validate_sides(
        side_a, side_b, n_per_side=REVIEWER_FIXED_SIZE, seeds=REVIEWER_SEEDS
    )
    full_row, words = gd.cross_validate_sides(
        side_a, side_b, seeds=REVIEWER_SEEDS, n_shuffles=20
    )
    row = {
        **description,
        "n_fixed": fixed_row["n_per_side"],
        "fixed_accuracy": fixed_row["cv_accuracy"],
        "fixed_accuracy_sd": fixed_row["cv_accuracy_sd"],
        "shuffled_accuracy": fixed_row["shuffled_accuracy"],
        "p_value": fixed_row["p_value"],
        "n_full": full_row["n_per_side"],
        "full_accuracy": full_row["cv_accuracy"],
    }
    return row, words.assign(**description)


def run_same_reviewer(named):
    """One reviewer, one grape, two countries. Writing style stays fixed, so what's left is the wine."""
    rows, words = [], []
    counts = named.groupby(["taster_name", "grape", "country"]).size()
    for (taster, grape), by_country in counts.groupby(level=["taster_name", "grape"]):
        countries = [
            country
            for (_, _, country), n in by_country.items()
            if n >= REVIEWER_MIN_REVIEWS and country in CONTINENT
        ]
        reviews = named[named["taster_name"].eq(taster)]
        for country_a, country_b in combinations(sorted(countries), 2):
            description = {
                "test": "same reviewer, two countries",
                "kind": near_or_far(country_a, country_b),
                "grape": grape,
                "reviewer_a": taster,
                "reviewer_b": taster,
                "place_a": country_a,
                "place_b": country_b,
            }
            try:
                row, pair_words = run_small_comparison(
                    gd.select_side(reviews, grape, country_a, "country"),
                    gd.select_side(reviews, grape, country_b, "country"),
                    description,
                )
            except ValueError as err:
                print("Skipped", description, err)
                continue
            rows.append(row)
            words.append(pair_words)
    return rows, words


def run_two_reviewers(named):
    """Two reviewers, one grape, one province. The wine stays fixed, so what's left is writing style."""
    rows, words = [], []
    counts = named.groupby(["grape", "province", "taster_name"]).size()
    for (grape, province), by_taster in counts.groupby(level=["grape", "province"]):
        tasters = [
            taster
            for (_, _, taster), n in by_taster.items()
            if n >= REVIEWER_MIN_REVIEWS
        ]
        reviews = named[named["grape"].eq(grape) & named["province"].eq(province)]
        for taster_a, taster_b in combinations(sorted(tasters), 2):
            description = {
                "test": "two reviewers, same place",
                "kind": "same place",
                "grape": grape,
                "reviewer_a": taster_a,
                "reviewer_b": taster_b,
                "place_a": province,
                "place_b": province,
            }
            try:
                row, pair_words = run_small_comparison(
                    reviews[reviews["taster_name"].eq(taster_a)],
                    reviews[reviews["taster_name"].eq(taster_b)],
                    description,
                )
            except ValueError as err:
                print("Skipped", description, err)
                continue
            rows.append(row)
            words.append(pair_words)
    return rows, words


def mixed_accuracy(reviews, grape_a, grape_b, province):
    """The same pair and place at the same size, with reviewers mixed, to compare with one reviewer."""
    row, _ = gd.cross_validate_sides(
        gd.select_side(reviews, grape_a, province),
        gd.select_side(reviews, grape_b, province),
        n_per_side=REVIEWER_FIXED_SIZE,
        seeds=REVIEWER_SEEDS,
        n_shuffles=1,
    )
    return row["cv_accuracy"]


def run_within_reviewer(df, named, progress):
    """
    One reviewer, one place, two grapes of the same color. Writing style and place both stay fixed,
    so what's left is the grape. This is the most direct test of whether reviews describe grapes differently.
    """
    color = df.drop_duplicates("grape").set_index("grape")["color"]
    usable = named[named["split"].ne("test")]
    counts = usable.groupby(["taster_name", "province", "grape"]).size()
    rows, words = [], []
    for (taster, province), by_grape in counts.groupby(level=["taster_name", "province"]):
        grapes = [
            grape
            for (_, _, grape), n in by_grape.items()
            if n >= REVIEWER_FIXED_SIZE
        ]
        reviews = named[named["taster_name"].eq(taster)]
        for grape_a, grape_b in combinations(sorted(grapes), 2):
            if color[grape_a] != color[grape_b]:
                continue
            description = {
                "test": "same reviewer, same place, two grapes",
                "kind": f"{color[grape_a]}s",
                "grape_a": grape_a,
                "grape_b": grape_b,
                "reviewer": taster,
                "place": province,
            }
            row, pair_words = run_small_comparison(
                gd.select_side(reviews, grape_a, province),
                gd.select_side(reviews, grape_b, province),
                description,
            )
            row["mixed_named_accuracy"] = mixed_accuracy(named, grape_a, grape_b, province)
            row["mixed_all_accuracy"] = mixed_accuracy(df, grape_a, grape_b, province)
            rows.append(row)
            words.append(pair_words)
            progress(f"{len(rows)}: {taster}, {province}, {grape_a} vs {grape_b}: {row['fixed_accuracy']}")

    # The control: there's nothing real to find, so these should land near 0.50.
    rng = np.random.default_rng(CONTROL_SEED)
    for (taster, province, grape), n in counts.items():
        if n < CONTROL_MIN_REVIEWS:
            continue
        reviews = gd.select_side(named[named["taster_name"].eq(taster)], grape, province)
        half = rng.random(len(reviews)) < 0.5
        fixed_row, _ = gd.cross_validate_sides(
            reviews[half],
            reviews[~half],
            n_per_side=REVIEWER_FIXED_SIZE,
            seeds=REVIEWER_SEEDS,
        )
        rows.append(
            {
                "test": "same reviewer, same grape, random split",
                "kind": "control",
                "grape_a": grape,
                "grape_b": grape,
                "reviewer": taster,
                "place": province,
                "n_fixed": fixed_row["n_per_side"],
                "fixed_accuracy": fixed_row["cv_accuracy"],
                "fixed_accuracy_sd": fixed_row["cv_accuracy_sd"],
                "shuffled_accuracy": fixed_row["shuffled_accuracy"],
                "p_value": fixed_row["p_value"],
            }
        )
    progress(f"{sum(r['kind'] == 'control' for r in rows)} controls done")

    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(WITHIN_SCORES_PATH, index=False)
    pd.concat(words).to_csv(WITHIN_WORDS_PATH, index=False)
    progress(f"Saved {len(rows)} tests to {WITHIN_SCORES_PATH}")


def main():
    start = time.time()

    def progress(message):
        print(f"[{time.time() - start:5.0f}s] {message}", flush=True)

    df = gd.load_reviews()
    if sys.argv[1:] == ["within_reviewer"]:
        run_within_reviewer(df, df[df["taster_name"].notna()], progress)
        return

    named = df[df["taster_name"].notna()]
    review_sets = {"all": df, "named_taster": named}

    scores, words, transfers = [], [], []
    for name, reviews in review_sets.items():
        for run in [run_planned, run_grids, run_near_far]:
            progress(f"{run.__name__} on {name}: {len(reviews):,} reviews")
            run_scores, run_words = run(reviews)
            scores.append(run_scores.assign(reviews_used=name))
            words.append(run_words.assign(reviews_used=name))
        progress(f"run_transfers on {name}")
        transfers.append(run_transfers(reviews).assign(reviews_used=name))

    os.makedirs(RESULTS, exist_ok=True)
    scores, words = pd.concat(scores), pd.concat(words)
    scores.to_csv(SCORES_PATH, index=False)
    words.to_csv(WORDS_PATH, index=False)
    pd.concat(transfers).to_csv(TRANSFER_PATH, index=False)
    progress(f"Saved {len(scores):,} comparisons and {sum(map(len, transfers)):,} transfers")

    progress("run_same_reviewer")
    same_rows, same_words = run_same_reviewer(named)
    progress(f"run_two_reviewers ({len(same_rows)} same-reviewer tests done)")
    two_rows, two_words = run_two_reviewers(named)
    pd.DataFrame(same_rows + two_rows).to_csv(REVIEWER_SCORES_PATH, index=False)
    pd.concat(same_words + two_words).to_csv(REVIEWER_WORDS_PATH, index=False)
    progress(f"Saved {len(same_rows)} same-reviewer and {len(two_rows)} two-reviewer tests")


if __name__ == "__main__":
    main()
