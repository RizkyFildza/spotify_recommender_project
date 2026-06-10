"""Entry point for the Spotify track recommendation pipeline.

Running ``python main.py`` with no arguments executes the full pipeline:

1. load and validate the raw dataset (reporting differences vs. the Kaggle
   dataset card),
2. clean the data and build the one-row-per-track catalogue,
3. generate EDA figures,
4. build the popularity baseline and the content-based recommender,
5. produce sample recommendations and save them to
   ``outputs/sample_recommendations.csv``,
6. run the proxy evaluation and save it to
   ``outputs/evaluation_results.csv``.

Ad-hoc queries are also supported, e.g.::

    python main.py --recommend "Smells Like Teen Spirit" --artist "Nirvana"
    python main.py --genres "jazz" --prefs "acousticness=0.85,energy=0.25"
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from src import config
from src.data_loading import dataset_overview, load_raw_data, print_overview
from src.eda import run_eda
from src.evaluation import evaluate_recommenders, print_evaluation
from src.preprocessing import build_catalogue, clean_data
from src.recommender import ContentBasedRecommender, PopularityRecommender
from src.utils import print_header, print_recommendations

# Demo queries used for the sample-recommendations output. Each was verified
# to exist in this version of the dataset; missing ones are skipped politely.
DEMO_TRACK_QUERIES = [
    ("Smells Like Teen Spirit", "Nirvana"),
    ("Shape of You", "Ed Sheeran"),
    ("Master of Puppets", "Metallica"),
]
DEMO_PROFILE = {
    "genres": ["jazz"],
    "preferences": {"acousticness": 0.85, "energy": 0.25},
}
DEMO_BASELINE_GENRE = "rock"


def build_recommenders(catalogue: pd.DataFrame) -> tuple[PopularityRecommender, ContentBasedRecommender]:
    """Construct both recommenders and fit the content-based model."""
    popularity_rec = PopularityRecommender(catalogue)
    start = time.perf_counter()
    content_rec = ContentBasedRecommender(catalogue).fit()
    print(
        f"Content-based index built over {len(catalogue):,} tracks "
        f"({content_rec._X.shape[1]} feature dimensions) "
        f"in {time.perf_counter() - start:.1f}s"
    )
    return popularity_rec, content_rec


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, and collapse the dataset; returns (cleaned, catalogue)."""
    raw = load_raw_data()
    cleaned = clean_data(raw, verbose=False)
    catalogue = build_catalogue(cleaned, verbose=False)
    return cleaned, catalogue


def run_track_query(content_rec: ContentBasedRecommender, track: str,
                    artist: str | None, n: int) -> pd.DataFrame | None:
    """Run a single track query, print it, and return the result frame."""
    try:
        query_row, n_matches, recs = content_rec.recommend(track, artist=artist, n=n)
    except LookupError as exc:
        print(f"  ! {exc}")
        return None
    matched = f"{query_row['track_name']} - {query_row['artists']}"
    note = f" (most popular of {n_matches} matching catalogue entries)" if n_matches > 1 else ""
    print(f"\nQuery track: {matched}{note}")
    print(f"  genres: {query_row['genres']} | popularity: {query_row['popularity']}/100")
    print_recommendations(recs, title=f"Top {len(recs)} similar tracks:")
    return recs


def run_profile_query(content_rec: ContentBasedRecommender, genres: list[str],
                      preferences: dict[str, float], n: int) -> pd.DataFrame | None:
    """Run a genre/preference profile query, print it, and return the frame."""
    try:
        profile, recs = content_rec.recommend_from_profile(
            genres, preferences=preferences, n=n
        )
    except ValueError as exc:
        print(f"  ! {exc}")
        return None
    print(f"\nProfile query: genres={profile['genres']}, preferences={profile['preferences']}")
    print_recommendations(recs, title=f"Top {len(recs)} tracks for this profile:")
    return recs


def run_full_pipeline(args: argparse.Namespace) -> None:
    print_header("1. Loading and validating the dataset")
    raw = load_raw_data()
    print_overview(dataset_overview(raw))

    print_header("2. Cleaning and building the track catalogue")
    cleaned = clean_data(raw, verbose=True)
    catalogue = build_catalogue(cleaned, verbose=True)

    if not args.skip_eda:
        print_header("3. Exploratory data analysis")
        run_eda(cleaned, catalogue)

    print_header("4. Building the recommenders")
    popularity_rec, content_rec = build_recommenders(catalogue)

    print_header("5. Sample recommendations")
    samples: list[pd.DataFrame] = []

    print("\n--- Baseline: most popular tracks " f"in '{DEMO_BASELINE_GENRE}' ---")
    baseline = popularity_rec.recommend(genre=DEMO_BASELINE_GENRE, n=5)
    print_recommendations(baseline)
    baseline.insert(0, "query", f"genre={DEMO_BASELINE_GENRE}")
    baseline.insert(0, "query_type", "popularity_baseline")
    samples.append(baseline)

    print("\n--- Content-based recommender ---")
    for track, artist in DEMO_TRACK_QUERIES:
        recs = run_track_query(content_rec, track, artist, n=args.top_n)
        if recs is not None:
            recs.insert(0, "query", f"{track} - {artist}")
            recs.insert(0, "query_type", "track")
            samples.append(recs)

    recs = run_profile_query(
        content_rec, DEMO_PROFILE["genres"], DEMO_PROFILE["preferences"], n=args.top_n
    )
    if recs is not None:
        recs.insert(0, "query", f"genres={DEMO_PROFILE['genres']}, prefs={DEMO_PROFILE['preferences']}")
        recs.insert(0, "query_type", "profile")
        samples.append(recs)

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(samples, ignore_index=True).to_csv(
        config.SAMPLE_RECOMMENDATIONS_PATH, index=False
    )
    print(f"\nSaved sample recommendations to {config.SAMPLE_RECOMMENDATIONS_PATH}")

    if not args.skip_eval:
        print_header("6. Proxy evaluation (no user interaction data available)")
        summary = evaluate_recommenders(catalogue, content_rec, popularity_rec)
        print_evaluation(summary, n_queries=config.EVAL_N_QUERIES, k=config.EVAL_TOP_K)
        print(f"\nSaved evaluation table to {config.EVALUATION_RESULTS_PATH}")

    print_header("Pipeline complete")
    print(f"Figures:                {config.FIGURES_DIR}")
    print(f"Sample recommendations: {config.SAMPLE_RECOMMENDATIONS_PATH}")
    print(f"Evaluation results:     {config.EVALUATION_RESULTS_PATH}")


def parse_prefs(prefs: str | None) -> dict[str, float]:
    """Parse '--prefs energy=0.3,acousticness=0.8' into a dict."""
    if not prefs:
        return {}
    parsed = {}
    for chunk in prefs.split(","):
        if "=" not in chunk:
            raise SystemExit(f"Could not parse preference '{chunk}'. Use feature=value.")
        key, value = chunk.split("=", 1)
        try:
            parsed[key.strip()] = float(value)
        except ValueError:
            raise SystemExit(f"Preference value for '{key.strip()}' must be numeric.")
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Content-based Spotify track recommender."
    )
    parser.add_argument("--recommend", metavar="TRACK",
                        help="recommend tracks similar to this track name")
    parser.add_argument("--artist", help="optional artist filter for --recommend")
    parser.add_argument("--genres",
                        help="comma-separated genres for a profile query, e.g. 'jazz,soul'")
    parser.add_argument("--prefs",
                        help="feature overrides for --genres, e.g. 'energy=0.3,acousticness=0.8'")
    parser.add_argument("-n", "--top-n", type=int, default=config.DEFAULT_TOP_N,
                        help=f"number of recommendations (default {config.DEFAULT_TOP_N})")
    parser.add_argument("--skip-eda", action="store_true",
                        help="skip figure generation in the full pipeline")
    parser.add_argument("--skip-eval", action="store_true",
                        help="skip the evaluation step in the full pipeline")
    args = parser.parse_args(argv)

    if args.recommend or args.genres:
        _, catalogue = load_and_prepare()
        _, content_rec = build_recommenders(catalogue)
        if args.recommend:
            run_track_query(content_rec, args.recommend, args.artist, n=args.top_n)
        if args.genres:
            genres = [g for g in args.genres.split(",")]
            run_profile_query(content_rec, genres, parse_prefs(args.prefs), n=args.top_n)
        return

    run_full_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
