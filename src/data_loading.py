"""Loading and basic validation of the raw Kaggle Spotify Tracks CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import config

# Columns the project relies on (the raw CSV additionally ships an
# unnamed index column, which is handled during preprocessing).
EXPECTED_COLUMNS = {
    "track_id", "artists", "album_name", "track_name", "popularity",
    "duration_ms", "explicit", "danceability", "energy", "key", "loudness",
    "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature", "track_genre",
}


def load_raw_data(path: Path = config.DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset and validate that all expected columns exist.

    Parameters
    ----------
    path:
        Location of the CSV. Defaults to ``data/dataset.csv`` inside the
        project folder.

    Returns
    -------
    pd.DataFrame
        The raw, unmodified dataset.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download the Spotify Tracks "
            "Dataset from Kaggle (maharshipandya/-spotify-tracks-dataset) "
            "and place it at data/dataset.csv."
        )

    df = pd.read_csv(path)

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset at '{path}' is missing expected columns: {sorted(missing)}"
        )
    return df


def dataset_overview(df: pd.DataFrame) -> dict:
    """Compute a compact overview of the raw dataset for reporting.

    The overview deliberately includes the checks used to verify the
    Kaggle description against the actual file (genre count, time-signature
    values, invalid tempo/duration rows, ...).
    """
    genre_counts = df["track_genre"].value_counts()
    overview = {
        "rows": len(df),
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "fully_duplicated_rows": int(df.duplicated().sum()),
        "unique_track_ids": int(df["track_id"].nunique()),
        "repeated_track_id_rows": int(len(df) - df["track_id"].nunique()),
        "unique_genres": int(df["track_genre"].nunique()),
        "rows_per_genre_min": int(genre_counts.min()),
        "rows_per_genre_max": int(genre_counts.max()),
        "rows_with_missing_metadata": int(
            df[["track_name", "artists", "album_name"]].isna().any(axis=1).sum()
        ),
        "rows_with_zero_tempo": int((df["tempo"] <= 0).sum()),
        "rows_with_zero_duration": int((df["duration_ms"] <= 0).sum()),
        "time_signature_values": sorted(df["time_signature"].unique().tolist()),
        "popularity_range": (int(df["popularity"].min()), int(df["popularity"].max())),
        "popularity_mean": float(df["popularity"].mean()),
        "popularity_median": float(df["popularity"].median()),
    }
    return overview


def print_overview(overview: dict) -> None:
    """Print the dataset overview, flagging differences vs. the Kaggle docs."""
    print(f"Rows: {overview['rows']:,} | Columns: {overview['columns']}")
    print(f"Fully duplicated rows: {overview['fully_duplicated_rows']}")
    print(
        f"Unique track_ids: {overview['unique_track_ids']:,} "
        f"({overview['repeated_track_id_rows']:,} rows are repeats of an id)"
    )
    print(
        f"Genres: {overview['unique_genres']} "
        f"(each with {overview['rows_per_genre_min']:,}-"
        f"{overview['rows_per_genre_max']:,} rows)"
    )
    print(
        f"Popularity: range {overview['popularity_range']}, "
        f"mean {overview['popularity_mean']:.2f}, "
        f"median {overview['popularity_median']:.0f}"
    )
    print(f"Rows with missing metadata: {overview['rows_with_missing_metadata']}")
    print(
        f"Rows with invalid audio values: tempo<=0 -> "
        f"{overview['rows_with_zero_tempo']}, duration<=0 -> "
        f"{overview['rows_with_zero_duration']}"
    )
    print(f"time_signature values present: {overview['time_signature_values']}")

    # Differences vs. the Kaggle dataset card, surfaced explicitly.
    notes = []
    if overview["unique_genres"] != 125:
        notes.append(
            f"Kaggle description mentions 125 genres; this file contains "
            f"{overview['unique_genres']}."
        )
    unexpected_ts = [t for t in overview["time_signature_values"] if t < 3 or t > 7]
    if unexpected_ts:
        notes.append(
            f"Kaggle documents time_signature in [3, 7], but values "
            f"{unexpected_ts} also occur."
        )
    if overview["rows_with_zero_tempo"] or overview["rows_with_zero_duration"]:
        notes.append(
            "Some rows have tempo = 0 and/or duration_ms = 0, which are not "
            "physically meaningful and are removed during cleaning."
        )
    if notes:
        print("\nDifferences vs. the Kaggle dataset card:")
        for note in notes:
            print(f"  - {note}")
