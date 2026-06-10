"""Small helper utilities shared across the project."""

from __future__ import annotations

import re

import pandas as pd


def normalise_text(value: str) -> str:
    """Lower-case a string and collapse internal whitespace for matching.

    Used to make track/artist search case- and spacing-insensitive.
    """
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip().casefold()


def primary_artist(artists: str) -> str:
    """Return the first (lead) artist from a semicolon-separated string."""
    if not isinstance(artists, str):
        return ""
    return artists.split(";")[0].strip()


def format_duration(duration_ms: float) -> str:
    """Format a duration in milliseconds as ``m:ss`` (e.g. 215000 -> '3:35')."""
    total_seconds = int(round(duration_ms / 1000))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def format_feature_value(feature: str, value: float) -> str:
    """Format a raw feature value for human-readable explanations."""
    if feature == "tempo":
        return f"{value:.0f} BPM"
    if feature == "loudness":
        return f"{value:.1f} dB"
    if feature == "duration_ms":
        return format_duration(value)
    return f"{value:.2f}"


def print_header(title: str, width: int = 78) -> None:
    """Print a visually distinct section header to the console."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_recommendations(result: pd.DataFrame, title: str = "") -> None:
    """Pretty-print a recommendation DataFrame to the console."""
    if title:
        print(f"\n{title}")
    if result.empty:
        print("  (no recommendations found)")
        return
    for _, row in result.iterrows():
        score = (
            f"sim {row['similarity_score']:.3f}"
            if "similarity_score" in row and row["similarity_score"] == row["similarity_score"]
            else f"popularity {row['popularity']}"
        )
        print(
            f"  {int(row['rank']):>2}. {row['track_name']} - {row['artists']}"
            f"  [{row['genres']}] ({score})"
        )
        if row.get("why_similar"):
            print(f"      ↳ {row['why_similar']}")