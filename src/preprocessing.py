"""Data cleaning and construction of the recommendation catalogue.

Cleaning decisions (each justified in README.md / report_summary.md):

1. Drop the ``Unnamed: 0`` column - it is only a row index from the CSV
   export and carries no information.
2. Drop the single row with missing ``track_name``/``artists``/``album_name``
   - a track that cannot be named cannot be searched for or recommended in a
   readable way.
3. Drop rows with ``tempo <= 0`` or ``duration_ms <= 0`` - these are failed
   feature extractions (157 + 1 rows, ~0.14% of the data) and would distort
   a distance-based recommender.
4. Repeated ``track_id`` values are NOT blindly removed. Inspection shows
   that for rows sharing a ``track_id``, every audio feature and metadata
   field is identical; only ``track_genre`` differs (the dataset samples
   1,000 rows per genre, so multi-genre tracks appear once per genre) and,
   for a small minority of ids, ``popularity`` (Spotify popularity is
   time-varying, so repeated API pulls can disagree slightly). The correct
   treatment is therefore to collapse each ``track_id`` to ONE catalogue row
   that keeps ALL of its genre labels and the maximum observed popularity.
"""

from __future__ import annotations

import pandas as pd

from src import config
from src.utils import normalise_text, primary_artist

METADATA_COLUMNS = ["track_name", "artists", "album_name"]


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply row/column-level cleaning to the raw dataset.

    Returns a cleaned DataFrame that still contains one row per
    (track_id, genre) pair; use :func:`build_catalogue` afterwards to get
    one row per track.
    """
    cleaned = df.copy()
    n_start = len(cleaned)

    # 1. Drop the exported index column.
    if "Unnamed: 0" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Unnamed: 0"])

    # 2. Drop rows with missing core metadata.
    missing_mask = cleaned[METADATA_COLUMNS].isna().any(axis=1)
    n_missing = int(missing_mask.sum())
    cleaned = cleaned[~missing_mask]

    # 3. Drop rows with physically impossible audio values.
    invalid_mask = (cleaned["tempo"] <= 0) | (cleaned["duration_ms"] <= 0)
    n_invalid = int(invalid_mask.sum())
    cleaned = cleaned[~invalid_mask]

    # Ensure consistent dtypes.
    cleaned["explicit"] = cleaned["explicit"].astype(bool)
    cleaned = cleaned.reset_index(drop=True)

    if verbose:
        print(f"Cleaning: started with {n_start:,} rows")
        print("  - dropped 'Unnamed: 0' index column")
        print(f"  - dropped {n_missing} row(s) with missing track/artist/album name")
        print(f"  - dropped {n_invalid} row(s) with tempo <= 0 or duration_ms <= 0")
        print(f"  -> {len(cleaned):,} rows remain (one per track-genre pair)")
    return cleaned


def build_catalogue(cleaned: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Collapse the cleaned data to one row per ``track_id``.

    Genres are aggregated into both a list column (``genre_list``) and a
    display string (``genres``); popularity takes the maximum observed value
    for the id. A safety check verifies the assumption that audio features
    are constant within a ``track_id`` before collapsing.
    """
    # Safety check: audio features must not vary within a track_id.
    feature_variation = (
        cleaned.groupby("track_id")[config.AUDIO_FEATURES].nunique().gt(1).any().any()
    )
    if feature_variation:
        print(
            "WARNING: some track_ids have inconsistent audio features; "
            "keeping the first occurrence of each."
        )

    agg_spec = {col: "first" for col in cleaned.columns if col not in ("track_id", "track_genre")}
    agg_spec["popularity"] = "max"  # popularity occasionally differs between pulls
    agg_spec["track_genre"] = lambda genres: sorted(set(genres))

    catalogue = (
        cleaned.groupby("track_id", as_index=False)
        .agg(agg_spec)
        .rename(columns={"track_genre": "genre_list"})
    )
    catalogue["genres"] = catalogue["genre_list"].str.join("; ")

    # Helper columns used for search, de-duplication and diversity control.
    catalogue["track_name_norm"] = catalogue["track_name"].map(normalise_text)
    catalogue["artists_norm"] = catalogue["artists"].map(normalise_text)
    catalogue["primary_artist"] = catalogue["artists"].map(primary_artist)
    catalogue["song_key"] = catalogue["track_name_norm"] + " || " + catalogue["artists_norm"]

    catalogue = catalogue.reset_index(drop=True)

    if verbose:
        n_multi = int((catalogue["genre_list"].str.len() > 1).sum())
        print(
            f"Catalogue: {len(catalogue):,} unique tracks "
            f"({n_multi:,} of them carry more than one genre label)"
        )
    return catalogue
