"""
Makes the charts for the presentation, saved to presentation/. Reads the saved results, so run
run_grape_differences.py first. Takes a few seconds:
    python make_presentation_charts.py

Word charts show only words that hold up: words at least two reviewers agree on, or words in the top
list on both all reviews and named-taster reviews. Label and filler words (riserva, imported, based)
are left out, since they don't describe how the wine tastes.
"""

import os
import textwrap

import grape_differences as gd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = "presentation"
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#ffffff", "#e5e4e0"
COLOR_A, COLOR_B = "#2a78d6", "#eb6834"
# One color per grape across every chart, so Chardonnay looks the same on every slide.
# Each pair shown together was checked for color-blind separation.
GRAPE_COLORS = {
    "Pinot Noir": "#2a78d6",
    "Chardonnay": "#eb6834",
    "Sauvignon Blanc": "#1baf7a",
    "Syrah": "#e34948",
    "Nebbiolo": "#e87ba4",
    "Sangiovese": "#008300",
    "Bordeaux-style Red Blend": "#6250d6",
    "Red Blend": "#eda100",
}
NOT_TASTE = {
    "cru", "riserva", "imported", "made", "made_entirely", "entirely", "soon", "good",
    "vineyard", "based", "five", "year", "best", "estate", "bit", "equal", "red_blend",
    "color", "alcohol", "texture", "variety", "varietal", "taste", "style", "mediterranean",
}

plt.rcParams.update({"font.size": 13, "axes.titlesize": 16, "figure.dpi": 100})


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def add_titles(ax, title, subtitle):
    """Title and a wrapped subtitle, lined up with the left edge of the plot."""
    subtitle = textwrap.fill(subtitle, 85)
    lines = subtitle.count("\n") + 1
    ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 10),
                textcoords="offset points", ha="left", va="bottom", fontsize=12, color=MUTED)
    ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(0, 18 + 17 * lines),
                textcoords="offset points", ha="left", va="bottom", fontsize=16, color=INK)


def word_share(side, terms):
    """% of a side's reviews (train and val only) that use each word."""
    tokens = side.loc[side["split"].ne("test"), "descriptor_ngrams"].str.split().apply(set)
    return pd.Series({term: sum(term in words for words in tokens) / len(tokens) * 100 for term in terms})


def plot_word_gaps(side_a, side_b, words_a, words_b, label_a, label_b, title, subtitle, colors, n=6):
    """
    One row per word: how often each side's reviews use it. The words above the line lean toward
    label_a, the ones below toward label_b. Each group keeps its n biggest gaps.
    """
    words_a = [w for w in words_a if w not in NOT_TASTE]
    words_b = [w for w in words_b if w not in NOT_TASTE]
    shares = pd.DataFrame({
        "a": word_share(side_a, words_a + words_b),
        "b": word_share(side_b, words_a + words_b),
    })
    shares["gap"] = shares["a"] - shares["b"]
    top_a = shares.loc[words_a].nlargest(n, "gap")
    top_b = shares.loc[words_b].nsmallest(n, "gap")
    rows = pd.concat([top_a, top_b.iloc[::-1]]).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 2.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = np.arange(len(rows))
    ax.hlines(y, rows[["a", "b"]].min(axis=1), rows[["a", "b"]].max(axis=1), color=GRID, linewidth=3, zorder=1)
    # Shape and fill tell the sides apart too, so the chart still works printed in black and white.
    ax.scatter(rows["a"], y, s=95, marker="o", color=colors[0], edgecolor=SURFACE, linewidth=1.5, zorder=3, label=label_a)
    ax.scatter(rows["b"], y, s=80, marker="s", facecolor=SURFACE, edgecolor=colors[1], linewidth=2.2, zorder=3, label=label_b)
    divider = len(top_b) - 0.5
    ax.axhline(divider, color=MUTED, linewidth=0.8, linestyle="--")
    side_note = {"fontsize": 11, "color": MUTED, "style": "italic", "ha": "right", "transform": ax.get_yaxis_transform()}
    ax.text(1.0, divider + 0.15, f"more common in {label_a} reviews  \u25b2", va="bottom", **side_note)
    ax.text(1.0, divider - 0.15, f"more common in {label_b} reviews  \u25bc", va="top", **side_note)
    ax.set_yticks(y, [term.replace("_", " ") for term in rows.index], color=INK)
    ax.set_xlabel("% of reviews that use the word", color=MUTED)
    ax.set_xlim(0, None)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.tick_params(axis="y", colors=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / len(rows)), ncol=2, frameon=False, labelcolor=INK)
    add_titles(ax, title, subtitle)
    return fig


def plot_pairs_by_reviewer(within, pairs, place, title):
    """Each pair's accuracy for each reviewer who covers it, same size and place."""
    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle="--")
    ax.text(0.5, len(pairs) - 0.45, "Guessing: 0.50", color=MUTED, ha="center", va="bottom")
    for row, (grape_a, grape_b) in enumerate(reversed(pairs)):
        tests = within[within["grape_a"].eq(grape_a) & within["grape_b"].eq(grape_b) & within["place"].eq(place)]
        ax.scatter(tests["fixed_accuracy"], [row] * len(tests), s=110, color=COLOR_A, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(1.02, row, f"average {tests['fixed_accuracy'].mean():.2f}", va="center", color=INK)
    ax.set_yticks(range(len(pairs)), [f"{a} vs {b}" for a, b in reversed(pairs)], color=INK)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.6, len(pairs) - 0.2)
    ax.set_xlabel("How often the model picks the right grape (80 reviews each, one reviewer)", color=MUTED)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.tick_params(axis="y", colors=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    reviewers = sorted(within.loc[within["place"].eq(place) & within["grape_a"].eq(pairs[0][0]) & within["grape_b"].eq(pairs[0][1]), "reviewer"])
    add_titles(ax, title, f"Each dot is one reviewer ({', '.join(reviewers)}), in {place}.")
    return fig


def agreed_words(within_words, grape_a, grape_b, place):
    """Words at least two reviewers lean the same way on, for one pair in one place."""
    agreement = gd.reviewer_agreement(within_words)
    row = agreement[agreement["grape_a"].eq(grape_a) & agreement["grape_b"].eq(grape_b) & agreement["place"].eq(place)].iloc[0]
    return row["agreed_for_a"].split(", "), row["agreed_for_b"].split(", ")


def steady_words(df, grape_a, grape_b, place, level):
    """Words in the top list in most of 5 samples, on both all reviews and named-taster reviews."""
    kept = {}
    for name, reviews in [("all", df), ("named", df[df["taster_name"].notna()])]:
        row, words = gd.compare_pair(reviews, grape_a, grape_b, place, level=level)
        kept[name] = {side: set(words.loc[words["favors"].eq(side) & words["in_top_runs"].ge(3), "term"]) for side in "ab"}
        kept[name]["accuracy"] = row["val_accuracy"]
    return [w for w in kept["all"]["a"] if w in kept["named"]["a"]], [w for w in kept["all"]["b"] if w in kept["named"]["b"]], kept["all"]["accuracy"]


def main():
    os.makedirs(OUT, exist_ok=True)
    df = gd.load_reviews()
    within = pd.read_csv("results/grape_within_reviewer_scores.csv")
    within_words = pd.read_csv("results/grape_within_reviewer_top_words.csv")
    small_tests = pd.read_csv("results/grape_same_reviewer_scores.csv")
    scores = pd.read_csv("results/grape_difference_scores.csv")

    def within_average(grape_a, grape_b, place):
        tests = within[within["grape_a"].eq(grape_a) & within["grape_b"].eq(grape_b) & within["place"].eq(place)]
        return tests["fixed_accuracy"].mean(), len(tests)

    # Slide 4: the trust check
    checks = gd.check_pair(df, "Chardonnay", "Sauvignon Blanc", place="California")
    save(gd.plot_check(checks, "Chardonnay vs Sauvignon Blanc: the real model vs shuffled labels"), "slide04_trust_check.png")
    print("  check:", checks["summary"][["accuracy", "accuracy_low", "accuracy_high", "shuffled_accuracy_mean", "shuffled_accuracy_max"]].to_dict())

    # Slide 5a: red vs white
    words_a, words_b, accuracy = steady_words(df, "Pinot Noir", "Chardonnay", "California", "province")
    save(plot_word_gaps(
        gd.select_side(df, "Pinot Noir", "California"), gd.select_side(df, "Chardonnay", "California"),
        words_a, words_b, "Pinot Noir (red)", "Chardonnay (white)",
        "Red vs white: Pinot Noir vs Chardonnay, California",
        f"Words that hold up on all reviews and named-taster reviews. The model is right {accuracy:.0%} of the time (guessing is 50%).",
        colors=(GRAPE_COLORS["Pinot Noir"], GRAPE_COLORS["Chardonnay"]),
    ), "slide05a_red_vs_white.png")

    # Slide 5b: two whites
    accuracy, n = within_average("Chardonnay", "Sauvignon Blanc", "California")
    words_a, words_b = agreed_words(within_words, "Chardonnay", "Sauvignon Blanc", "California")
    save(plot_word_gaps(
        gd.select_side(df, "Chardonnay", "California"), gd.select_side(df, "Sauvignon Blanc", "California"),
        words_a, words_b, "Chardonnay", "Sauvignon Blanc",
        "Chardonnay vs Sauvignon Blanc, California",
        f"Words reviewers agree on. With one reviewer at a time, the model is right {accuracy:.0%} of the time ({n} reviewers; guessing is 50%).",
        colors=(GRAPE_COLORS["Chardonnay"], GRAPE_COLORS["Sauvignon Blanc"]),
    ), "slide05b_chardonnay_vs_sauvignon_blanc.png")

    # Slide 6: two reds, easy vs hard
    accuracy, n = within_average("Pinot Noir", "Syrah", "California")
    words_a, words_b = agreed_words(within_words, "Pinot Noir", "Syrah", "California")
    save(plot_word_gaps(
        gd.select_side(df, "Pinot Noir", "California"), gd.select_side(df, "Syrah", "California"),
        words_a, words_b, "Pinot Noir", "Syrah",
        "Pinot Noir vs Syrah, California",
        f"Words reviewers agree on. With one reviewer at a time, the model is right {accuracy:.0%} of the time ({n} reviewers; guessing is 50%).",
        colors=(GRAPE_COLORS["Pinot Noir"], GRAPE_COLORS["Syrah"]),
    ), "slide06_pinot_noir_vs_syrah.png")
    save(plot_pairs_by_reviewer(
        within,
        [("Pinot Noir", "Syrah"), ("Chardonnay", "Sauvignon Blanc"), ("Cabernet Sauvignon", "Merlot")],
        "California",
        "Same reviewers, same place: Cabernet and Merlot barely separate",
    ), "slide06_pairs_by_reviewer.png")

    # Slide 7: Nebbiolo vs Sangiovese on the clean text
    words_a, words_b, accuracy = steady_words(df, "Nebbiolo", "Sangiovese", "Italy", "country")
    italy = df[df["country"].eq("Italy")]
    save(plot_word_gaps(
        gd.select_side(italy, "Nebbiolo"), gd.select_side(italy, "Sangiovese"),
        words_a, words_b, "Nebbiolo", "Sangiovese",
        "Nebbiolo vs Sangiovese, Italy",
        f"Words that hold up on all reviews and named-taster reviews. The model is right {accuracy:.0%} of the time (guessing is 50%).",
        colors=(GRAPE_COLORS["Nebbiolo"], GRAPE_COLORS["Sangiovese"]),
    ), "slide07_nebbiolo_vs_sangiovese.png")

    # Slide 9: writing style vs grape vs country
    within["test"] = within["test"].fillna("")
    save(gd.plot_small_tests(pd.concat([small_tests, within]), "Who wrote it separates reviews more than the grape or the country"), "slide09_what_separates_reviews.png")

    # Optional: US red blends on the clean text
    processed = pd.read_parquet(gd.INPUT_PATH, columns=["variety", "province", "split", "taster_name", "descriptor_ngrams"])
    california = processed[processed["province"].eq("California")]
    bordeaux_style = california[california["variety"].eq("Bordeaux-style Red Blend")]
    red_blend = california[california["variety"].eq("Red Blend")]
    accuracies, word_runs = [], []
    for seed in range(5):
        train, val = gd.sample_sides(bordeaux_style, red_blend, seed=seed)
        model = gd.fit_pair(train)
        accuracies.append(gd.score_pair(model, val)["accuracy"])
        word_runs.append(gd.top_words(model, train, 20))
    runs = pd.concat(word_runs).groupby(["favors", "term"]).size()
    steady = {side: list(runs[side][runs[side].ge(3)].index) for side in "ab"}
    save(plot_word_gaps(
        bordeaux_style, red_blend, steady["a"], steady["b"], "Bordeaux-style Red Blend", "Red Blend",
        "Bordeaux-style Red Blend vs Red Blend, California",
        f"Words in the top list in most of 5 samples. The model is right {np.mean(accuracies):.0%} of the time (guessing is 50%).",
        colors=(GRAPE_COLORS["Bordeaux-style Red Blend"], GRAPE_COLORS["Red Blend"]),
    ), "optional_red_blends.png")

    planned = scores[scores["run"].eq("planned") & scores["kind"].eq("red vs white")]
    print("  Pinot Noir vs Chardonnay:", planned[["reviews_used", "val_accuracy"]].to_dict("records"))
    print("  Cabernet vs Merlot, one reviewer:", within_average("Cabernet Sauvignon", "Merlot", "California"))


if __name__ == "__main__":
    main()
