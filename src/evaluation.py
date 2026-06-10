"""Proxy evaluation of the recommenders.

The dataset contains no user interaction data (no plays, likes or
playlists), so classical accuracy metrics such as precision@k against held
-out listens are impossible. Instead, three systems are compared on proxy
metrics over a fixed random sample of query tracks:

* ``random``             - k tracks drawn uniformly at random (floor).
* ``popularity_global``  - the global top-k most popular tracks
                           (the non-personalised baseline).
* ``content_based``      - the proposed model.

Metrics (averaged over queries, higher/lower as indicated):

* ``genre_overlap@k``       (higher better) - share of recommendations that
  have at least one genre in common with the query track.
* ``audio_distance@k``      (lower better) - mean Euclidean distance between
  query and recommendations in *unweighted* z-scored audio space. Using the
  unweighted space keeps the metric from trivially favouring the model that
  was tuned on the weighted space.
* ``artist_diversity@k``    (higher better) - unique primary artists divided
  by k; guards against degenerate "10 tracks by the same artist" lists.
* ``mean_popularity@k``     (descriptive) - average popularity of the
  recommended tracks; shows each system's popularity bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.recommender import ContentBasedRecommender, PopularityRecommender

METRIC_COLUMNS = [
    "genre_overlap@k", "audio_distance@k", "artist_diversity@k", "mean_popularity@k",
]


def evaluate_recommenders(
    catalogue: pd.DataFrame,
    content_rec: ContentBasedRecommender,
    popularity_rec: PopularityRecommender,
    n_queries: int = config.EVAL_N_QUERIES,
    k: int = config.EVAL_TOP_K,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """Run the proxy evaluation and return a summary table (one row per system)."""
    rng = np.random.default_rng(seed)
    n_items = len(catalogue)
    query_indices = rng.choice(n_items, size=min(n_queries, n_items), replace=False)

    genre_sets = [set(gl) for gl in catalogue["genre_list"]]
    artist_keys = catalogue["primary_artist"].str.casefold().to_numpy()
    popularity = catalogue["popularity"].to_numpy()
    audio = content_rec.audio_scaled  # unweighted z-scores

    # Global popularity list is identical for every query; fetch k+1 so the
    # query track itself can be removed when it happens to be in the top-k.
    popularity_top = popularity_rec.top_indices(k + 1)

    per_query: dict[str, list[list[float]]] = {
        "random": [], "popularity_global": [], "content_based": [],
    }

    for q in query_indices:
        q = int(q)
        random_recs = [int(r) for r in rng.choice(n_items, size=k + 1, replace=False) if r != q][:k]
        popularity_recs = [r for r in popularity_top if r != q][:k]
        content_recs = content_rec.recommend_indices(q, k)

        for system, recs in (
            ("random", random_recs),
            ("popularity_global", popularity_recs),
            ("content_based", content_recs),
        ):
            if not recs:
                continue
            overlap = float(np.mean([bool(genre_sets[q] & genre_sets[r]) for r in recs]))
            distance = float(np.linalg.norm(audio[recs] - audio[q], axis=1).mean())
            diversity = len({artist_keys[r] for r in recs}) / len(recs)
            mean_pop = float(popularity[recs].mean())
            per_query[system].append([overlap, distance, diversity, mean_pop])

    summary = pd.DataFrame(
        {system: np.mean(values, axis=0) for system, values in per_query.items()},
        index=METRIC_COLUMNS,
    ).T
    summary.index.name = "system"
    summary = summary.round(3)

    config.EVALUATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(config.EVALUATION_RESULTS_PATH)
    return summary


def print_evaluation(summary: pd.DataFrame, n_queries: int, k: int) -> None:
    """Print the evaluation table with a short reading guide."""
    print(f"Proxy evaluation over {n_queries} random query tracks (top-{k}):\n")
    print(summary.to_string())
    print(
        "\nHow to read this: genre_overlap and artist_diversity are better "
        "when higher; audio_distance is better when lower; mean_popularity "
        "is descriptive (it exposes each system's popularity bias)."
    )
