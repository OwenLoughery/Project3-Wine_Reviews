"""
Functions for explaining the differences between grapes can be used by calling in grape_differences.ipynb.

Each comparison is one small model: TF-IDF + logistic regression on two sides, where a side is a
grape (optionally from one place). Both sides get the same number of reviews, so a bigger grape
doesn't win just by having more data and scores can be compared across pairs. The weights say which
words lean toward each side.

We can then use these to compare grapes like red vs. white, two reds that are similar,
two grapes from the same country/area, etc.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline

INPUT_PATH = "data/processed_reviews.parquet"
COLUMNS = [
    "review_id",
    "grape",
    "color",
    "country",
    "province",
    "split",
    "taster_name",
    "in_target",
    "label_type",
    "descriptor_ngrams",
    "clean_ngrams",
]

# descriptor_ngrams has no grape or place names, so what's left is how the wine gets described
TEXT_COL = "descriptor_ngrams"
SPLIT_SHARE = {"train": 0.70, "val": 0.15, "test": 0.15}
ALL_COUNTRIES = "All countries"


def load_reviews(path=INPUT_PATH):
    """Loads only the columns we need and keeps the single-grape target classes."""
    df = pd.read_parquet(path, columns=COLUMNS)
    return df[df["in_target"] & df["label_type"].eq("grape")].reset_index(drop=True)


def select_side(df, grape, place=None, level="province"):
    """One grape, from one place if given. level is the column the place is in: province or country."""
    keep = df["grape"].eq(grape)
    if place is not None:
        keep &= df[level].eq(place)
    return df[keep]


def sample_sides(side_a, side_b, n_per_side=None, seed=0):
    """
    Returns train and val reviews for two sides with equal counts, and side = 1 for side_a.

    n_per_side fixes the total per side so a grid of pairs is comparable; by default the bigger side
    is cut down to the size of the smaller one. The test split is left alone on purpose.
    """
    parts = {}
    for split in ["train", "val"]:
        a, b = side_a[side_a["split"].eq(split)], side_b[side_b["split"].eq(split)]
        n = min(len(a), len(b))
        if n_per_side is not None:
            wanted = round(n_per_side * SPLIT_SHARE[split])
            if n < wanted:
                raise ValueError(
                    f"only {n} {split} reviews on the smaller side, wanted {wanted}"
                )
            n = wanted
        parts[split] = pd.concat(
            [
                a.sample(n, random_state=seed).assign(side=1),
                b.sample(n, random_state=seed).assign(side=0),
            ],
            ignore_index=True,
        )
    return parts["train"], parts["val"]


def sample_pair(
    df,
    grape_a,
    grape_b,
    place=None,
    place_b=None,
    level="province",
    n_per_side=None,
    seed=0,
):
    """
    Same as sample_sides, for two grapes. place_b lets one grape be compared across places
    (Pinot Noir: California vs Burgundy).
    """
    side_a = select_side(df, grape_a, place, level)
    side_b = select_side(df, grape_b, place if place_b is None else place_b, level)
    return sample_sides(side_a, side_b, n_per_side, seed)


def fit_pair(train, text_col=TEXT_COL, min_df=3, C=1.0):
    """The bigrams are already in the text as black_cherry, so TF-IDF only needs single tokens."""
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 1), min_df=min_df, sublinear_tf=True),
            ),
            ("clf", LogisticRegression(C=C, max_iter=1000)),
        ]
    )
    return model.fit(train[text_col], train["side"])


def score_pair(model, reviews, text_col=TEXT_COL):
    """Accuracy and AUC. The sides are the same size, so guessing gets 0.50 on both."""
    return {
        "accuracy": accuracy_score(reviews["side"], model.predict(reviews[text_col])),
        "auc": roc_auc_score(
            reviews["side"], model.decision_function(reviews[text_col])
        ),
    }


def top_words(model, train, n=20, text_col=TEXT_COL):
    """
    The n words leaning hardest toward each side, with how often each shows up on each side
    (pct_a and pct_b are % of that side's training reviews), which is easier to explain than a weight.
    """
    terms = model.named_steps["tfidf"].get_feature_names_out()
    weights = model.named_steps["clf"].coef_[0]

    present = model.named_steps["tfidf"].transform(train[text_col]) > 0
    is_a = train["side"].eq(1).to_numpy()
    pct_a = np.asarray(present[is_a].mean(axis=0)).ravel() * 100
    pct_b = np.asarray(present[~is_a].mean(axis=0)).ravel() * 100

    words = pd.DataFrame(
        {"term": terms, "weight": weights, "pct_a": pct_a, "pct_b": pct_b}
    )
    return pd.concat(
        [
            words.nlargest(n, "weight").assign(favors="a"),
            words.nsmallest(n, "weight").assign(favors="b"),
        ],
        ignore_index=True,
    )


def top_value(side, column):
    """The most common value on a side (a taster, a country) and its share of the rows that have one."""
    values = side[column].dropna()
    if values.empty:
        return None, np.nan
    shares = values.value_counts(normalize=True)
    return shares.index[0], round(float(shares.iloc[0]), 3)


def compare_pair(
    df,
    grape_a,
    grape_b,
    place=None,
    place_b=None,
    level="province",
    n_per_side=None,
    seeds=range(5),
    n_words=20,
    text_col=TEXT_COL,
    min_df=3,
    C=1.0,
):
    """
    Runs one comparison over several random samples. Returns one row of scores and a table of words.
    in_top_runs counts how many samples put a word in the top list, so a word that only shows up once
    can be treated as noise.
    """
    place_b = place if place_b is None else place_b
    scores, word_runs = [], []
    for seed in seeds:
        train, val = sample_pair(
            df, grape_a, grape_b, place, place_b, level, n_per_side, seed
        )
        model = fit_pair(train, text_col, min_df, C)
        train_score, val_score = (
            score_pair(model, train, text_col),
            score_pair(model, val, text_col),
        )
        scores.append(
            {
                "train_accuracy": train_score["accuracy"],
                "val_accuracy": val_score["accuracy"],
                "val_auc": val_score["auc"],
            }
        )
        word_runs.append(top_words(model, train, n_words, text_col))

    scores = pd.DataFrame(scores)
    side_a = select_side(df, grape_a, place, level)
    side_b = select_side(df, grape_b, place_b, level)
    reviewer_a, reviewer_share_a = top_value(side_a, "taster_name")
    reviewer_b, reviewer_share_b = top_value(side_b, "taster_name")
    country_a, country_share_a = top_value(side_a, "country")
    country_b, country_share_b = top_value(side_b, "country")
    place_a_label = ALL_COUNTRIES if place is None else place
    place_b_label = ALL_COUNTRIES if place_b is None else place_b
    row = {
        "grape_a": grape_a,
        "grape_b": grape_b,
        "level": level if place is not None else "all",
        "place_a": place_a_label,
        "place_b": place_b_label,
        "n_train_per_side": len(train) // 2,
        "n_val_per_side": len(val) // 2,
        **scores.mean().round(3).to_dict(),
        "val_accuracy_sd": round(float(scores["val_accuracy"].std()), 3),
        "top_reviewer_a": reviewer_a,
        "top_reviewer_a_share": reviewer_share_a,
        "top_reviewer_b": reviewer_b,
        "top_reviewer_b_share": reviewer_share_b,
        "same_top_reviewer": reviewer_a == reviewer_b and reviewer_a is not None,
        "top_country_a": country_a,
        "top_country_a_share": country_share_a,
        "top_country_b": country_b,
        "top_country_b_share": country_share_b,
    }

    words = (
        pd.concat(word_runs)
        .groupby(["favors", "term"], as_index=False)
        .agg(
            in_top_runs=("term", "size"),
            weight=("weight", "mean"),
            pct_a=("pct_a", "mean"),
            pct_b=("pct_b", "mean"),
        )
        .assign(
            favors_grape=lambda d: d["favors"].map({"a": grape_a, "b": grape_b}),
            strength=lambda d: d["weight"].abs(),
        )
        .sort_values(
            ["favors", "in_top_runs", "strength"], ascending=[True, False, False]
        )
        .drop(columns="strength")
    )
    words = words.assign(
        grape_a=grape_a, grape_b=grape_b, place_a=place_a_label, place_b=place_b_label
    )
    return row, words.round(3).reset_index(drop=True)


def side_label(grape, place, other_place):
    """Just the grape, unless the two sides are the same grape from different places."""
    return grape if place == other_place else f"{grape} ({place})"


def check_pair(
    df,
    grape_a,
    grape_b,
    place=None,
    place_b=None,
    level="province",
    n_per_side=None,
    seed=0,
    n_shuffles=200,
    n_boot=1000,
    n_words=20,
    text_col=TEXT_COL,
    min_df=3,
    C=1.0,
):
    """
    How much to trust one comparison, on the validation reviews of one sample. Returns a dict with:

    summary: accuracy and AUC with 95% ranges from resampling the validation reviews, the accuracy
        when the training labels are shuffled (what luck alone gets), and the accuracy using only
        the top n_words per side (whether the word lists really explain the model).
    confusion: rows are the true grape, columns are what the model said.
    by_grape: precision, recall and F1 for each side.
    shuffled_accuracy: the accuracy from every shuffle, for plotting.
    """
    place_b = place if place_b is None else place_b
    train, val = sample_pair(df, grape_a, grape_b, place, place_b, level, n_per_side, seed)
    model = fit_pair(train, text_col, min_df, C)
    tfidf = model.named_steps["tfidf"]
    X_train, X_val = tfidf.transform(train[text_col]), tfidf.transform(val[text_col])
    y_train, y_val = train["side"].to_numpy(), val["side"].to_numpy()
    predicted = model.predict(val[text_col])
    margin = model.decision_function(val[text_col])
    rng = np.random.default_rng(seed)

    # the 95% range: score the same predictions on resampled validation reviews
    rows = rng.integers(0, len(val), size=(n_boot, len(val)))
    boot_accuracy = (y_val[rows] == predicted[rows]).mean(axis=1)
    boot_auc = [roc_auc_score(y_val[r], margin[r]) for r in rows]

    # what luck gets: same text, shuffled labels. TF-IDF never sees the labels so it can be reused.
    shuffled_accuracy = np.array(
        [
            LogisticRegression(C=C, max_iter=1000)
            .fit(X_train, rng.permutation(y_train))
            .score(X_val, y_val)
            for _ in range(n_shuffles)
        ]
    )
    accuracy = accuracy_score(y_val, predicted)

    # picked from the training reviews only, so this doesn't get to peek at validation
    top = top_words(model, train, n_words, text_col)["term"]
    columns = [tfidf.vocabulary_[term] for term in top]
    top_words_accuracy = (
        LogisticRegression(C=C, max_iter=1000)
        .fit(X_train[:, columns], y_train)
        .score(X_val[:, columns], y_val)
    )

    summary = {
        "val_reviews": len(val),
        "accuracy": accuracy,
        "accuracy_low": np.percentile(boot_accuracy, 2.5),
        "accuracy_high": np.percentile(boot_accuracy, 97.5),
        "auc": roc_auc_score(y_val, margin),
        "auc_low": np.percentile(boot_auc, 2.5),
        "auc_high": np.percentile(boot_auc, 97.5),
        "shuffled_accuracy_mean": shuffled_accuracy.mean(),
        "shuffled_accuracy_max": shuffled_accuracy.max(),
        "p_value": ((shuffled_accuracy >= accuracy).sum() + 1) / (n_shuffles + 1),
        "top_words_used": len(columns),
        "top_words_accuracy": top_words_accuracy,
    }

    label_a = side_label(grape_a, place, place_b)
    label_b = side_label(grape_b, place_b, place)
    names = [label_a, label_b]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, predicted, labels=[1, 0]
    )
    return {
        "summary": pd.Series(summary).round(3),
        "confusion": pd.DataFrame(
            confusion_matrix(y_val, predicted, labels=[1, 0]),
            index=[f"true {name}" for name in names],
            columns=[f"predicted {name}" for name in names],
        ),
        "by_grape": pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "reviews": support},
            index=names,
        ).round(3),
        "shuffled_accuracy": shuffled_accuracy,
    }


def plot_check(checks, title):
    """The real accuracy and its 95% range next to what shuffled labels get, from check_pair."""
    summary = checks["summary"]
    ink, muted, surface, shuffled_color, real_color = (
        "#0b0b0b",
        "#52514e",
        "#fcfcfb",
        "#2a78d6",
        "#eb6834",
    )

    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=surface)
    ax.set_facecolor(surface)
    ax.hist(
        checks["shuffled_accuracy"],
        bins=np.arange(0.40, 1.0, 0.01),
        color=shuffled_color,
        edgecolor=surface,
        linewidth=2,
    )
    ax.axvspan(
        summary["accuracy_low"],
        summary["accuracy_high"],
        color=real_color,
        alpha=0.15,
        linewidth=0,
    )
    # dark so it stands apart from the bars when printed in black and white
    ax.axvline(summary["accuracy"], color=ink, linewidth=2.5)

    top = ax.get_ylim()[1]
    ax.text(
        summary["shuffled_accuracy_mean"],
        top * 1.02,
        f"Shuffled labels: {summary['shuffled_accuracy_mean']:.2f} on average",
        color=ink,
        ha="center",
        va="bottom",
    )
    ax.text(
        summary["accuracy"],
        top * 1.02,
        f"Real model: {summary['accuracy']:.3f}\n"
        f"95% range {summary['accuracy_low']:.3f} to {summary['accuracy_high']:.3f}",
        color=ink,
        ha="center",
        va="bottom",
    )

    ax.set_title(title, loc="left", color=ink, fontsize=13, pad=44)
    ax.set_xlim(0.40, max(0.85, summary["accuracy_high"] + 0.08))
    ax.set_xlabel("Validation accuracy", color=muted)
    ax.set_ylabel("Shuffled runs", color=muted)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.tick_params(colors=muted, length=0)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def transfer_pair(
    df,
    grape_a,
    grape_b,
    place_train,
    place_test,
    level="country",
    n_per_side=300,
    seeds=range(5),
    text_col=TEXT_COL,
    min_df=3,
    C=1.0,
):
    """
    Does a model travel? Trains on one place and scores it on another place's validation reviews,
    next to a model trained in that other place on the same number of reviews. Both get scored on
    the same reviews, so the gap between travel_accuracy and local_accuracy is what the move costs.
    """
    rows = []
    for seed in seeds:
        home_train, home_val = sample_pair(
            df, grape_a, grape_b, place_train, place_train, level, n_per_side, seed
        )
        away_train, away_val = sample_pair(
            df, grape_a, grape_b, place_test, place_test, level, n_per_side, seed
        )
        home_model = fit_pair(home_train, text_col, min_df, C)
        away_model = fit_pair(away_train, text_col, min_df, C)
        rows.append(
            {
                "home_accuracy": score_pair(home_model, home_val, text_col)["accuracy"],
                "travel_accuracy": score_pair(home_model, away_val, text_col)["accuracy"],
                "local_accuracy": score_pair(away_model, away_val, text_col)["accuracy"],
            }
        )
    means = pd.DataFrame(rows).mean()

    def reviewer_at(place):
        reviews = df[df["grape"].isin([grape_a, grape_b]) & df[level].eq(place)]
        return top_value(reviews, "taster_name")[0]

    return {
        "grape_a": grape_a,
        "grape_b": grape_b,
        "trained_in": place_train,
        "tested_in": place_test,
        "n_train_per_side": len(home_train) // 2,
        "n_val_per_side": len(away_val) // 2,
        **means.round(3).to_dict(),
        "cost_of_moving": round(
            float(means["local_accuracy"] - means["travel_accuracy"]), 3
        ),
        "same_top_reviewer": reviewer_at(place_train) == reviewer_at(place_test),
    }


def cross_validate_sides(
    side_a,
    side_b,
    n_per_side=None,
    seeds=range(5),
    n_shuffles=100,
    n_words=15,
    text_col=TEXT_COL,
    min_df=2,
    C=1.0,
):
    """
    For comparisons too small for a fixed validation set, like one reviewer's Riesling from two
    countries. Uses 5-fold cross-validation on the train and val reviews, leaving test alone. The
    p-value comes from shuffling the labels. Returns one row of scores and the top words.
    """
    side_a = side_a[side_a["split"].ne("test")]
    side_b = side_b[side_b["split"].ne("test")]
    n = min(len(side_a), len(side_b)) if n_per_side is None else n_per_side
    if n > min(len(side_a), len(side_b)):
        raise ValueError(f"only {min(len(side_a), len(side_b))} reviews, wanted {n}")

    accuracies, shuffled, p_values, word_runs = [], [], [], []
    for seed in seeds:
        reviews = pd.concat(
            [
                side_a.sample(n, random_state=seed).assign(side=1),
                side_b.sample(n, random_state=seed).assign(side=0),
            ],
            ignore_index=True,
        )
        accuracy, shuffled_scores, p_value = permutation_test_score(
            Pipeline(
                [
                    ("tfidf", TfidfVectorizer(min_df=min_df, sublinear_tf=True)),
                    ("clf", LogisticRegression(C=C, max_iter=1000)),
                ]
            ),
            reviews[text_col],
            reviews["side"],
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            n_permutations=n_shuffles,
            random_state=seed,
        )
        accuracies.append(accuracy)
        shuffled.append(shuffled_scores.mean())
        p_values.append(p_value)
        word_runs.append(
            top_words(fit_pair(reviews, text_col, min_df, C), reviews, n_words, text_col)
        )

    row = {
        "n_per_side": n,
        "cv_accuracy": round(float(np.mean(accuracies)), 3),
        "cv_accuracy_sd": round(float(np.std(accuracies, ddof=1)), 3),
        "shuffled_accuracy": round(float(np.mean(shuffled)), 3),
        "p_value": round(float(np.median(p_values)), 3),
    }
    words = (
        pd.concat(word_runs)
        .groupby(["favors", "term"], as_index=False)
        .agg(
            in_top_runs=("term", "size"),
            weight=("weight", "mean"),
            pct_a=("pct_a", "mean"),
            pct_b=("pct_b", "mean"),
        )
        .assign(strength=lambda d: d["weight"].abs())
        .sort_values(["favors", "in_top_runs", "strength"], ascending=[True, False, False])
        .drop(columns="strength")
        .round(3)
    )
    return row, words


def look_up_pair(scores, words, grape_a, grape_b, place, place_b=None, run="planned"):
    """
    Pulls one comparison out of the saved results. in_both is True when a word made the top list in
    most samples on all reviews and on named-taster reviews. Those are the words to trust.
    """
    most_samples = 3 if run == "planned" else 2
    place_b = place if place_b is None else place_b

    def this_pair(table):
        return table[
            table["run"].eq(run)
            & table["grape_a"].eq(grape_a)
            & table["grape_b"].eq(grape_b)
            & table["place_a"].eq(place)
            & table["place_b"].eq(place_b)
        ]

    pair_scores = this_pair(scores)
    if pair_scores.empty:
        raise ValueError(f"no saved {run} run for {grape_a} vs {grape_b}, {place}")

    pair_words = this_pair(words)
    label_a = side_label(grape_a, place, place_b)
    label_b = side_label(grape_b, place_b, place)
    table = (
        pair_words.pivot_table(
            index=["favors", "term"],
            columns="reviews_used",
            values=["in_top_runs", "pct_a", "pct_b"],
        )
        .reindex(columns=["all", "named_taster"], level=1)
        .pipe(lambda d: d.set_axis([f"{a}_{b}" for a, b in d.columns], axis=1))
        .reset_index()
    )
    table["in_both"] = table["in_top_runs_all"].ge(most_samples) & table[
        "in_top_runs_named_taster"
    ].ge(most_samples)
    table["favors"] = table["favors"].map({"a": label_a, "b": label_b})
    table = table.sort_values(
        ["favors", "in_both", "in_top_runs_all", "pct_a_all"],
        ascending=[True, False, False, False],
    ).rename(
        columns={
            "in_top_runs_all": "runs_all",
            "in_top_runs_named_taster": "runs_named",
            "pct_a_all": f"% of {label_a}",
            "pct_b_all": f"% of {label_b}",
        }
    )
    columns = ["favors", "term", "in_both", "runs_all", "runs_named"]
    return pair_scores, table[columns + [f"% of {label_a}", f"% of {label_b}"]].round(1)


def grape_distinctiveness(scores, place, reviews_used="all"):
    """Each grape's average accuracy against the other grapes of its color in one place. Higher means easier to spot."""
    same_color = scores[
        scores["run"].eq("grid")
        & scores["place_a"].eq(place)
        & scores["kind"].isin(["reds", "whites"])
        & scores["reviews_used"].eq(reviews_used)
    ]
    sides = pd.concat(
        [
            same_color.rename(columns={side: "grape"})[["grape", "kind", "val_accuracy"]]
            for side in ["grape_a", "grape_b"]
        ]
    )
    return (
        sides.groupby(["kind", "grape"])["val_accuracy"]
        .agg(pairs="size", average_accuracy="mean", lowest="min", highest="max")
        .round(3)
        .sort_values(["kind", "average_accuracy"], ascending=[True, False])
        .reset_index()
    )


def score_matrix(scores, kind, reviews_used, place):
    """Pair scores as a grape-by-grape table, filled in on both sides of the diagonal."""
    pairs = scores[
        scores["run"].eq("grid")
        & scores["kind"].eq(kind)
        & scores["reviews_used"].eq(reviews_used)
        & scores["place_a"].eq(place)
    ]
    matrix = pairs.pivot(index="grape_a", columns="grape_b", values="val_accuracy")
    grapes = sorted(set(pairs["grape_a"]) | set(pairs["grape_b"]))
    matrix = matrix.reindex(index=grapes, columns=grapes)
    return matrix.combine_first(matrix.T)


def plot_heatmaps(scores, kind, place):
    """All reviews next to named-taster reviews, on one color scale and in one grape order."""
    ink, muted, surface, missing = "#0b0b0b", "#52514e", "#fcfcfb", "#ecebe7"
    ramp = LinearSegmentedColormap.from_list(
        "accuracy", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
    )
    ramp.set_bad(missing)

    panels = {
        "All reviews": score_matrix(scores, kind, "all", place),
        "Named-taster reviews": score_matrix(scores, kind, "named_taster", place),
    }
    # hardest-to-separate grapes first, so similar grapes end up near each other
    order = panels["All reviews"].mean().sort_values().index

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5.6), facecolor=surface, sharey=True, layout="constrained"
    )
    for ax, (title, matrix) in zip(axes, panels.items(), strict=True):
        matrix = matrix.reindex(index=order, columns=order)
        image = ax.imshow(matrix, cmap=ramp, vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(order)), order, rotation=45, ha="right", color=muted)
        ax.set_yticks(range(len(order)), order, color=muted)
        ax.set_title(title, loc="left", color=ink, fontsize=12)
        ax.tick_params(length=0)
        ax.spines[:].set_visible(False)
        # the 2px gap between cells
        ax.set_xticks(np.arange(len(order) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(order) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=surface, linewidth=2)
        ax.tick_params(which="minor", length=0)

    axes[1].tick_params(labelleft=False)

    bar = fig.colorbar(image, ax=axes, shrink=0.7, pad=0.02)
    bar.set_label("Validation accuracy (0.50 is chance)", color=muted)
    bar.ax.tick_params(colors=muted, length=0)
    bar.outline.set_visible(False)
    fig.suptitle(
        f"How easy it is to tell {kind} apart, {place}",
        x=0.01,
        ha="left",
        color=ink,
        fontsize=14,
    )
    return fig


def reviewer_agreement(within_words):
    """
    For grape pairs covered by two or more reviewers in one place: the words at least two of them
    independently lean the same way on, and how many words different reviewers lean opposite ways on.
    """
    steady = within_words[within_words["in_top_runs"].ge(2)]
    rows = []
    for (place, grape_a, grape_b), pair in steady.groupby(["place", "grape_a", "grape_b"]):
        if pair["reviewer"].nunique() < 2:
            continue
        reviewers = pair.groupby(["favors", "term"])["reviewer"].nunique()
        shared = reviewers[reviewers.ge(2)].sort_values(ascending=False)
        agreed = {
            side: ", ".join(term for favors, term in shared.index if favors == side)
            for side in ["a", "b"]
        }

        rows.append(
            {
                "place": place,
                "grape_a": grape_a,
                "grape_b": grape_b,
                "reviewers": pair["reviewer"].nunique(),
                "agreed_for_a": agreed["a"],
                "agreed_for_b": agreed["b"],
                "words_agreed": int(reviewers.ge(2).sum()),
                "words_opposite": int(pair.groupby("term")["favors"].nunique().eq(2).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_small_tests(small_tests, title):
    """One dot per small test (same-reviewer and within-reviewer results combined), all at the same size."""
    ink, muted, surface, dot = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
    groups = {
        "Two reviewers,\nsame grape, same place": small_tests["test"].str.startswith("two"),
        "Same reviewer, same place,\ntwo grapes": small_tests["kind"].isin(["reds", "whites"]),
        "Same reviewer, same grape,\ncountries on different continents": small_tests["kind"].eq("far"),
        "Same reviewer, same grape,\ncountries on the same continent": small_tests["kind"].eq("near"),
        "Same reviewer, same grape,\nsplit at random (control)": small_tests["kind"].eq("control"),
    }
    rng = np.random.default_rng(4401)

    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=surface, layout="constrained")
    ax.set_facecolor(surface)
    ax.axvline(0.5, color=muted, linewidth=1, linestyle="--")
    ax.text(0.5, 4.62, "Guessing: 0.50", color=muted, ha="center", va="bottom")
    for position, keep in zip([4, 3, 2, 1, 0], groups.values(), strict=True):
        scores = small_tests.loc[keep, "fixed_accuracy"]
        ax.scatter(
            scores,
            position + rng.uniform(-0.18, 0.18, len(scores)),
            s=80,
            color=dot,
            edgecolor=surface,
            linewidth=2,
            zorder=3,
        )
        ax.text(
            1.01,
            position,
            f"average {scores.mean():.2f}\n{len(scores)} tests",
            color=ink,
            va="center",
        )
    ax.set_yticks([4, 3, 2, 1, 0], list(groups), color=ink)
    ax.set_xlim(0.40, 1.0)
    ax.set_ylim(-0.5, 4.6)
    ax.set_xlabel("Cross-validated accuracy (80 reviews per side)", color=muted)
    ax.set_title(title, loc="left", color=ink, fontsize=13, pad=22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(muted)
    ax.tick_params(colors=muted, length=0)
    ax.tick_params(axis="y", colors=ink)
    ax.grid(axis="x", color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    return fig


def region_pairs(df, place=None, level="province", min_reviews=200, same_color=True):
    """Every pair of grapes in a place where both have at least min_reviews, optionally same color only."""
    reviews = df if place is None else df[df[level].eq(place)]
    counts = reviews.groupby(["grape", "color"]).size().reset_index(name="n")
    big = counts[counts["n"] >= min_reviews]
    color = dict(zip(big["grape"], big["color"], strict=True))
    pairs = combinations(sorted(color), 2)
    return [
        (a, b, place) for a, b in pairs if not same_color or color[a] == color[b]
    ]


def run_grid(
    df,
    pairs,
    level="province",
    n_per_side=200,
    seeds=range(3),
    n_words=20,
    text_col=TEXT_COL,
):
    """Runs compare_pair on (grape_a, grape_b, place) pairs. Pairs without enough reviews get skipped and reported."""
    rows, words, skipped = [], [], []
    for grape_a, grape_b, place in pairs:
        try:
            row, pair_words = compare_pair(
                df,
                grape_a,
                grape_b,
                place,
                level=level,
                n_per_side=n_per_side,
                seeds=seeds,
                n_words=n_words,
                text_col=text_col,
            )
        except ValueError as err:
            skipped.append((grape_a, grape_b, place, str(err)))
            continue
        rows.append(row)
        words.append(pair_words)
    if skipped:
        print(f"Skipped {len(skipped)} pairs without enough reviews:", skipped[:3])
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    return pd.DataFrame(rows), pd.concat(words, ignore_index=True)
