"""Central configuration for the Spotify track recommender.

Every path, feature list, and tunable parameter lives here so the rest of
the codebase stays free of hard-coded values. Changing the behaviour of the
system (e.g. feature weights, top-N size) should only require edits to this
file.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (resolved relative to the project root, so the project can be run
# from any working directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SAMPLE_RECOMMENDATIONS_PATH = OUTPUTS_DIR / "sample_recommendations.csv"
EVALUATION_RESULTS_PATH = OUTPUTS_DIR / "evaluation_results.csv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 2002

# ---------------------------------------------------------------------------
# Feature design
# ---------------------------------------------------------------------------
# Numerical audio features describing how a track *sounds*. These form the
# core of the similarity space.
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]

# Per-feature weights, applied AFTER standardisation (z-scoring). All audio
# features start on an equal footing (weight 1.0); a few are down-weighted
# with an explicit rationale:
#   - liveness:    audience noise says little about whether two songs are a
#                  good musical match.
#   - loudness:    strongly correlated with energy (r ~ 0.76 in this data),
#                  so full weight would double-count intensity.
#   - duration_ms: track length matters less than how a track sounds.
# Popularity is deliberately EXCLUDED from the similarity space: it measures
# how often a track is played, not what it sounds like, and including it
# would bias the recommender towards "similarly famous" rather than
# "similar sounding" tracks.
FEATURE_WEIGHTS = {
    "danceability": 1.0,
    "energy": 1.0,
    "loudness": 0.7,
    "speechiness": 1.0,
    "acousticness": 1.0,
    "instrumentalness": 1.0,
    "liveness": 0.5,
    "valence": 1.0,
    "tempo": 0.8,
    "duration_ms": 0.4,
}

# Weight of the (multi-hot) genre block relative to the audio block. The
# scaled audio block of 10 z-scored features has a typical vector norm of
# roughly sqrt(10) ~ 3.2, so a genre weight of 2.0 makes genre a strong
# secondary signal without letting it dominate the audio similarity.
GENRE_WEIGHT = 2.0

# Human-readable labels used when explaining recommendations.
FEATURE_LABELS = {
    "danceability": "danceability",
    "energy": "energy",
    "loudness": "loudness",
    "speechiness": "speechiness",
    "acousticness": "acousticness",
    "instrumentalness": "instrumentalness",
    "liveness": "liveness",
    "valence": "mood (valence)",
    "tempo": "tempo",
    "duration_ms": "track length",
}

# ---------------------------------------------------------------------------
# Recommendation behaviour
# ---------------------------------------------------------------------------
DEFAULT_TOP_N = 10          # recommendations returned per query
NEIGHBOUR_POOL_SIZE = 300   # candidates fetched before filtering/diversity
MAX_PER_ARTIST = 2          # diversity cap: max tracks per primary artist
EXPLANATION_FEATURES = 3    # number of audio features cited per explanation

# ---------------------------------------------------------------------------
# Evaluation settings (proxy evaluation; no user interaction data exists)
# ---------------------------------------------------------------------------
EVAL_N_QUERIES = 200        # number of random query tracks sampled
EVAL_TOP_K = 10             # recommendations evaluated per query
