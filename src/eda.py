"""Exploratory data analysis: a small set of purposeful figures.

Four figures are produced (kept deliberately minimal):

1. ``popularity_distribution.png``  - shape of the popularity score.
2. ``tracks_per_genre.png``         - genre balance check (raw rows vs.
                                      unique tracks after de-duplication).
3. ``feature_correlation_heatmap.png`` - relationships between audio
                                      features (motivates feature weights).
4. ``genre_audio_profiles.png``     - mean audio profile of contrasting
                                      genres (shows the features carry
                                      genre-relevant signal).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved, never shown

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

# Perceptual 0-1 features used for the genre profile heatmap.
PROFILE_FEATURES = [
    "danceability", "energy", "valence",
    "acousticness", "instrumentalness", "speechiness",
]
# A deliberately contrasting set of genres for the profile figure.
PROFILE_GENRES = [
    "classical", "sleep", "acoustic", "jazz",
    "pop", "hip-hop", "edm", "death-metal",
]


def run_eda(
    cleaned: pd.DataFrame,
    catalogue: pd.DataFrame,
    figures_dir: Path = config.FIGURES_DIR,
) -> None:
    """Generate all EDA figures and print a brief console summary."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    _plot_popularity_distribution(catalogue, figures_dir)
    _plot_genre_counts(cleaned, catalogue, figures_dir)
    _plot_feature_correlations(catalogue, figures_dir)
    _plot_genre_profiles(cleaned, figures_dir)

    print(f"Saved 4 figures to {figures_dir}")
    zero_pop = (catalogue["popularity"] == 0).mean()
    print(
        f"Note: {zero_pop:.1%} of unique tracks have popularity 0 - the "
        "catalogue has a long tail of obscure tracks, which is exactly why a "
        "popularity-only baseline is a weak recommender."
    )


def _plot_popularity_distribution(catalogue: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(catalogue["popularity"], bins=50, color="#1DB954", edgecolor="white")
    mean = catalogue["popularity"].mean()
    ax.axvline(mean, color="black", linestyle="--", linewidth=1, label=f"mean = {mean:.1f}")
    ax.set_xlabel("Popularity (0-100)")
    ax.set_ylabel("Number of unique tracks")
    ax.set_title("Popularity distribution across the catalogue")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "popularity_distribution.png", dpi=150)
    plt.close(fig)


def _plot_genre_counts(
    cleaned: pd.DataFrame, catalogue: pd.DataFrame, figures_dir: Path
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

    raw_counts = cleaned["track_genre"].value_counts().sort_index()
    ax1.bar(range(len(raw_counts)), raw_counts.values, color="#1DB954", width=1.0)
    ax1.set_xticks([])
    ax1.set_ylabel("Rows in raw data")
    ax1.set_title(
        f"Genre balance check: {len(raw_counts)} genres × ~1,000 rows each "
        "(small dips reflect removed invalid rows)"
    )

    unique_counts = (
        catalogue.explode("genre_list")["genre_list"].value_counts().sort_values()
    )
    ax2.bar(range(len(unique_counts)), unique_counts.values, color="#535353", width=1.0)
    ax2.set_xticks([])
    ax2.set_xlabel(
        "Genres (sorted) - after de-duplication, counts vary because "
        "multi-genre tracks are stored once"
    )
    ax2.set_ylabel("Unique tracks per genre")

    fig.tight_layout()
    fig.savefig(figures_dir / "tracks_per_genre.png", dpi=150)
    plt.close(fig)


def _plot_feature_correlations(catalogue: pd.DataFrame, figures_dir: Path) -> None:
    cols = config.AUDIO_FEATURES + ["popularity"]
    corr = catalogue[cols].corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="vlag", center=0,
        square=True, cbar_kws={"shrink": 0.8}, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation between audio features (and popularity)")
    fig.tight_layout()
    fig.savefig(figures_dir / "feature_correlation_heatmap.png", dpi=150)
    plt.close(fig)


def _plot_genre_profiles(cleaned: pd.DataFrame, figures_dir: Path) -> None:
    genres = [g for g in PROFILE_GENRES if g in set(cleaned["track_genre"])]
    profile = (
        cleaned[cleaned["track_genre"].isin(genres)]
        .groupby("track_genre")[PROFILE_FEATURES]
        .mean()
        .loc[genres]
    )
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.heatmap(
        profile, annot=True, fmt=".2f", cmap="YlGnBu",
        vmin=0, vmax=1, cbar_kws={"shrink": 0.9}, ax=ax,
    )
    ax.set_title("Mean audio profile of contrasting genres")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(figures_dir / "genre_audio_profiles.png", dpi=150)
    plt.close(fig)
