"""
Makes the figures for the written report, saved to report_figures/. One word chart per comparison in
the report, all in the same style, readable in color and printed in black and white. Run
run_grape_differences.py first (for the reviewer results), then:
    python make_report_figures.py

Every word chart uses the same rules: grape and place names removed (descriptor_ngrams), equal
reviews per side, and only words that hold up. For the comparisons we ran with single reviewers,
that means words at least two reviewers agree on. For the rest, it means words in the top list in
at least 3 of 5 random samples, on all reviews and on named-taster reviews.

Markers stay with the wine: a wine that shows up in more than one figure (Chardonnay, the two blends)
always has the same marker.
"""

import os

import grape_differences as gd
import matplotlib.pyplot as plt
import pandas as pd
from make_presentation_charts import COLOR_A, COLOR_B, agreed_words, plot_word_gaps

OUT = "report_figures"
SAMPLES = range(5)
# Blend reviews often just state the recipe ("a blend of", "equal parts", "the majority is Merlot"),
# and "red" mostly comes from the phrase "red blend". These say what's on the label, not how it tastes.
RECIPE_WORDS = {
    "blend", "combine", "equal", "equal_part", "majority", "make", "made", "five", "lead", "part",
    "percent", "red", "appellation", "bottling", "vineyard", "vine",
}
STEADY_SUBTITLE = (
    "Words in the top list in at least 3 of 5 random samples, on all reviews and on named-taster "
    "reviews. Grape and place names removed."
)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def steady_words(side_a, side_b):
    """Words in the top list in at least 3 of 5 samples, on all reviews and on named-taster reviews."""
    kept = []
    for named_only in [False, True]:
        a, b = side_a, side_b
        if named_only:
            a, b = a[a["taster_name"].notna()], b[b["taster_name"].notna()]
        runs = []
        for seed in SAMPLES:
            train, _ = gd.sample_sides(a, b, seed=seed)
            runs.append(gd.top_words(gd.fit_pair(train), train, 20))
        counts = pd.concat(runs).groupby(["favors", "term"]).size()
        kept.append({side: set(counts[side][counts[side].ge(3)].index) for side in "ab"})
    return [sorted(kept[0][side] & kept[1][side]) for side in "ab"]


def word_figure(name, side_a, side_b, label_a, label_b, title, subtitle, words=None, exclude=frozenset()):
    words_a, words_b = words if words is not None else steady_words(side_a, side_b)
    words_a = [w for w in words_a if w not in exclude]
    words_b = [w for w in words_b if w not in exclude]
    save(
        plot_word_gaps(side_a, side_b, words_a, words_b, label_a, label_b, title, subtitle, colors=(COLOR_A, COLOR_B)),
        name,
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    reviews = pd.read_parquet(
        gd.INPUT_PATH,
        columns=["grape", "variety", "country", "province", "split", "taster_name", "in_target", "label_type", "descriptor_ngrams"],
    )
    grapes = reviews[reviews["in_target"] & reviews["label_type"].eq("grape")]
    within_words = pd.read_csv("results/grape_within_reviewer_top_words.csv")

    def grape_in(grape, column, place):
        return grapes[grapes["grape"].eq(grape) & grapes[column].eq(place)]

    def variety_in(variety, province):
        return reviews[reviews["variety"].eq(variety) & reviews["province"].eq(province)]

    # Owen: red vs white
    word_figure(
        "fig_red_vs_white_chardonnay_vs_pinot_noir.png",
        grape_in("Chardonnay", "province", "California"), grape_in("Pinot Noir", "province", "California"),
        "Chardonnay (white)", "Pinot Noir (red)",
        "Red vs white: Chardonnay vs Pinot Noir, California", STEADY_SUBTITLE,
    )

    # Owen: whites vs whites, with the words at least two reviewers agree on
    word_figure(
        "fig_whites_chardonnay_vs_sauvignon_blanc.png",
        grape_in("Chardonnay", "province", "California"), grape_in("Sauvignon Blanc", "province", "California"),
        "Chardonnay", "Sauvignon Blanc",
        "Chardonnay vs Sauvignon Blanc, California",
        "Words at least two of the three California reviewers (Gordon, Kettmann, Boone) independently lean "
        "the same way on. Grape and place names removed.",
        words=agreed_words(within_words, "Chardonnay", "Sauvignon Blanc", "California"),
    )
    word_figure(
        "fig_whites_chardonnay_vs_riesling.png",
        grape_in("Chardonnay", "province", "Washington"), grape_in("Riesling", "province", "Washington"),
        "Chardonnay", "Riesling",
        "Chardonnay vs Riesling, Washington",
        "Words both Washington reviewers (Gregutt, Sullivan) independently lean the same way on. "
        "Grape and place names removed.",
        words=agreed_words(within_words, "Chardonnay", "Riesling", "Washington"),
    )
    checks = gd.check_pair(grapes, "Chardonnay", "Sauvignon Blanc", place="California")
    save(gd.plot_check(checks, "Chardonnay vs Sauvignon Blanc: the real model vs 200 shuffled-label models"),
         "fig_whites_trust_check.png")

    # Similar reds: Nebbiolo vs Sangiovese
    word_figure(
        "fig_similar_reds_nebbiolo_vs_sangiovese.png",
        grape_in("Nebbiolo", "country", "Italy"), grape_in("Sangiovese", "country", "Italy"),
        "Nebbiolo", "Sangiovese",
        "Nebbiolo vs Sangiovese, Italy", STEADY_SUBTITLE,
    )

    # Blends, California. Bordeaux-style always comes first and Red Blend always second, so each
    # keeps its marker across figures.
    word_figure(
        "fig_blends_bordeaux_style_vs_cabernet_sauvignon.png",
        variety_in("Bordeaux-style Red Blend", "California"), variety_in("Cabernet Sauvignon", "California"),
        "Bordeaux-style Red Blend", "Cabernet Sauvignon",
        "Bordeaux-style Red Blend vs Cabernet Sauvignon, California",
        STEADY_SUBTITLE + " Words about the blend recipe (blend, equal parts, majority) also removed. "
        "Only these taste words hold up: apart from the recipe, reviewers describe the two almost the same way.",
        exclude=RECIPE_WORDS,
    )
    word_figure(
        "fig_blends_zinfandel_vs_red_blend.png",
        variety_in("Zinfandel", "California"), variety_in("Red Blend", "California"),
        "Zinfandel", "Red Blend",
        "Zinfandel vs Red Blend, California",
        STEADY_SUBTITLE + " Words about the blend recipe (blend, combine) also removed.",
        exclude=RECIPE_WORDS,
    )
    word_figure(
        "fig_blends_bordeaux_style_vs_red_blend.png",
        variety_in("Bordeaux-style Red Blend", "California"), variety_in("Red Blend", "California"),
        "Bordeaux-style Red Blend", "Red Blend",
        "Bordeaux-style Red Blend vs Red Blend, California",
        STEADY_SUBTITLE + " Words about the blend recipe also removed.",
        exclude=RECIPE_WORDS,
    )


if __name__ == "__main__":
    main()
