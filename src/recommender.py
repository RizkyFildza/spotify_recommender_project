"""Recommendation models.

Two recommenders are implemented:

* :class:`PopularityRecommender` - a deliberately simple baseline that ranks
  tracks by Spotify popularity, optionally within a genre. It ignores audio
  content entirely and serves as the comparison point.

* :class:`ContentBasedRecommender` - the main model. Each track is embedded
  as the concatenation of (a) z-scored, weighted audio features and (b) a
  weighted multi-hot genre block. Similar tracks are retrieved with cosine
  nearest neighbours, filtered for duplicates and artist diversity, and each
  recommendation carries a human-readable explanation.

This is content-based filtering: the dataset contains no user listening
histories, so collaborative filtering is not possible here.
"""

from __future__ import annotations

import difflib

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from src import config
from src.utils import format_duration, format_feature_value, normalise_text

RESULT_COLUMNS = [
    "rank", "track_id", "track_name", "artists", "album_name", "genres",
    "popularity", "duration", "explicit", "similarity_score", "why_similar",
    "method",
]


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
class PopularityRecommender:
    """Popularity-based baseline: most-played tracks, globally or per genre."""

    def __init__(self, catalogue: pd.DataFrame):
        self.catalogue = catalogue.reset_index(drop=True)

    def _ranked(self, genre: str | None) -> pd.DataFrame:
        cat = self.catalogue
        if genre is not None:
            g = genre.strip().casefold()
            mask = cat["genre_list"].map(lambda genre_list: g in genre_list)
            cat = cat[mask]
            if cat.empty:
                raise ValueError(f"Genre '{genre}' does not exist in the catalogue.")
        # De-duplicate same-song variants (e.g. single vs. album release).
        return cat.sort_values("popularity", ascending=False).drop_duplicates("song_key")

    def top_indices(self, k: int, genre: str | None = None) -> list[int]:
        """Catalogue indices of the top-k most popular tracks (for evaluation)."""
        return self._ranked(genre).head(k).index.tolist()

    def recommend(self, genre: str | None = None, n: int = config.DEFAULT_TOP_N) -> pd.DataFrame:
        scope = f"in the '{genre}' genre" if genre else "overall"
        rows = []
        for rank, (_, row) in enumerate(self._ranked(genre).head(n).iterrows(), start=1):
            rows.append({
                "rank": rank,
                "track_id": row["track_id"],
                "track_name": row["track_name"],
                "artists": row["artists"],
                "album_name": row["album_name"],
                "genres": row["genres"],
                "popularity": row["popularity"],
                "duration": format_duration(row["duration_ms"]),
                "explicit": row["explicit"],
                "similarity_score": np.nan,
                "why_similar": f"#{rank} most popular track {scope} "
                               f"(popularity {row['popularity']}/100)",
                "method": "popularity_baseline",
            })
        return pd.DataFrame(rows, columns=RESULT_COLUMNS)


# ---------------------------------------------------------------------------
# Content-based model
# ---------------------------------------------------------------------------
class ContentBasedRecommender:
    """Explainable content-based recommender over audio + genre features."""

    def __init__(
        self,
        catalogue: pd.DataFrame,
        audio_features: list[str] | None = None,
        feature_weights: dict[str, float] | None = None,
        genre_weight: float | None = None,
    ):
        self.catalogue = catalogue.reset_index(drop=True)
        self.audio_features = list(audio_features or config.AUDIO_FEATURES)
        weights = feature_weights or config.FEATURE_WEIGHTS
        self.weight_vector = np.array(
            [weights[f] for f in self.audio_features], dtype=np.float32
        )
        self.genre_weight = config.GENRE_WEIGHT if genre_weight is None else genre_weight
        self._fitted = False

    # -- fitting ------------------------------------------------------------
    def fit(self) -> "ContentBasedRecommender":
        """Build the feature matrix and the nearest-neighbour index."""
        raw = self.catalogue[self.audio_features].to_numpy(dtype=np.float64)
        self.scaler = StandardScaler().fit(raw)

        # Unweighted z-scores are kept separately: they feed explanations
        # and the evaluation module (a weight-free notion of audio distance).
        self.audio_scaled = self.scaler.transform(raw).astype(np.float32)
        audio_block = self.audio_scaled * self.weight_vector

        # Multi-hot genre block, row-normalised so that a track with many
        # genre labels does not get a larger genre footprint than a track
        # with one label, then weighted as a secondary signal.
        self.mlb = MultiLabelBinarizer(sparse_output=True)
        genre_block = self.mlb.fit_transform(self.catalogue["genre_list"]).astype(np.float32)
        label_counts = np.asarray(genre_block.sum(axis=1)).ravel()
        row_norm = sparse.diags(1.0 / np.sqrt(np.maximum(label_counts, 1.0)))
        genre_block = (row_norm @ genre_block) * self.genre_weight

        self._X = sparse.hstack(
            [sparse.csr_matrix(audio_block), genre_block], format="csr"
        ).astype(np.float32)
        self._nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(self._X)

        self._genre_sets = [set(gl) for gl in self.catalogue["genre_list"]]
        self._song_keys = self.catalogue["song_key"].tolist()
        self._artist_keys = self.catalogue["primary_artist"].str.casefold().tolist()
        # Features eligible for explanations (down-weighted ones excluded).
        self._explain_features = [
            f for f, w in zip(self.audio_features, self.weight_vector) if w >= 0.7
        ] or self.audio_features
        self._feature_pos = {f: i for i, f in enumerate(self.audio_features)}
        self._fitted = True
        return self

    # -- search -------------------------------------------------------------
    def find_track(self, track_name: str, artist: str | None = None) -> tuple[int, int]:
        """Locate a track in the catalogue by name (and optionally artist).

        Exact (normalised) name matches are preferred over substring matches;
        among multiple matches the most popular one is chosen. Returns the
        catalogue index of the chosen track and the number of matches found.
        """
        name_norm = normalise_text(track_name)
        if not name_norm:
            raise ValueError("Please provide a non-empty track name.")
        artist_norm = normalise_text(artist) if artist else None

        cat = self.catalogue
        for matcher in (
            cat["track_name_norm"] == name_norm,                       # exact
            cat["track_name_norm"].str.contains(name_norm, regex=False),  # substring
        ):
            mask = matcher
            if artist_norm:
                mask = mask & cat["artists_norm"].str.contains(artist_norm, regex=False)
            matches = cat[mask]
            if not matches.empty:
                best = matches.sort_values("popularity", ascending=False).index[0]
                return int(best), int(len(matches))

        suffix = f" by '{artist}'" if artist else ""
        raise LookupError(
            f"No track matching '{track_name}'{suffix} was found in the "
            "catalogue. Check the spelling, drop the artist filter, or try a "
            "shorter part of the title."
        )

    # -- recommendation -----------------------------------------------------
    def recommend(
        self,
        track_name: str,
        artist: str | None = None,
        n: int = config.DEFAULT_TOP_N,
        max_per_artist: int | None = config.MAX_PER_ARTIST,
    ) -> tuple[pd.Series, int, pd.DataFrame]:
        """Recommend tracks similar to a named track.

        Returns ``(query_row, n_matches, recommendations)`` so the caller can
        show which catalogue entry was matched.
        """
        idx, n_matches = self.find_track(track_name, artist)
        return self.catalogue.loc[idx], n_matches, self.recommend_for_index(
            idx, n=n, max_per_artist=max_per_artist
        )

    def recommend_for_index(
        self,
        idx: int,
        n: int = config.DEFAULT_TOP_N,
        max_per_artist: int | None = config.MAX_PER_ARTIST,
    ) -> pd.DataFrame:
        """Recommend tracks similar to the catalogue entry at ``idx``."""
        self._check_fitted()
        pairs = self._neighbour_candidates(
            self._X[idx], n=n, max_per_artist=max_per_artist,
            exclude_index=idx, exclude_song_key=self._song_keys[idx],
        )
        return self._build_result(
            pairs,
            ref_scaled=self.audio_scaled[idx],
            ref_raw=self.catalogue.loc[idx, self.audio_features],
            ref_genres=self._genre_sets[idx],
        )

    def recommend_indices(self, idx: int, k: int) -> list[int]:
        """Bare list of recommended catalogue indices (used by evaluation)."""
        self._check_fitted()
        pairs = self._neighbour_candidates(
            self._X[idx], n=k, max_per_artist=config.MAX_PER_ARTIST,
            exclude_index=idx, exclude_song_key=self._song_keys[idx],
        )
        return [i for i, _ in pairs]

    def recommend_from_profile(
        self,
        genres: list[str],
        preferences: dict[str, float] | None = None,
        n: int = config.DEFAULT_TOP_N,
        max_per_artist: int | None = config.MAX_PER_ARTIST,
    ) -> tuple[dict, pd.DataFrame]:
        """Recommend tracks from a genre/preference profile (no seed track).

        The query vector is the mean audio profile of the requested genres,
        optionally overridden by explicit feature preferences such as
        ``{"energy": 0.25, "acousticness": 0.85}``.
        """
        self._check_fitted()
        genres = [g.strip().casefold() for g in genres if g.strip()]
        if not genres:
            raise ValueError("Please provide at least one genre.")
        known = set(self.mlb.classes_)
        unknown = [g for g in genres if g not in known]
        if unknown:
            hints = {
                g: difflib.get_close_matches(g, sorted(known), n=3) for g in unknown
            }
            raise ValueError(f"Unknown genre(s) {unknown}. Closest matches: {hints}")

        # Mean raw audio profile of tracks carrying any of the genres.
        member_idx = [i for i, s in enumerate(self._genre_sets) if s & set(genres)]
        profile_raw = self.catalogue.loc[member_idx, self.audio_features].mean()

        preferences = preferences or {}
        bad_keys = [k for k in preferences if k not in self.audio_features]
        if bad_keys:
            raise ValueError(
                f"Unknown preference feature(s) {bad_keys}. "
                f"Valid options: {self.audio_features}"
            )
        for feature, value in preferences.items():
            profile_raw[feature] = float(value)

        ref_scaled = self.scaler.transform(
            profile_raw.to_numpy(dtype=np.float64).reshape(1, -1)
        ).astype(np.float32)
        audio_block = ref_scaled * self.weight_vector

        genre_vec = self.mlb.transform([genres]).astype(np.float32)
        genre_vec = genre_vec * (self.genre_weight / np.sqrt(len(genres)))

        query_vec = sparse.hstack(
            [sparse.csr_matrix(audio_block), genre_vec], format="csr"
        )
        pairs = self._neighbour_candidates(
            query_vec, n=n, max_per_artist=max_per_artist,
            exclude_index=None, exclude_song_key=None,
        )
        result = self._build_result(
            pairs,
            ref_scaled=ref_scaled[0],
            ref_raw=profile_raw,
            ref_genres=set(genres),
        )
        profile_description = {
            "genres": genres,
            "preferences": preferences,
            "profile": {f: round(float(profile_raw[f]), 3) for f in self.audio_features},
        }
        return profile_description, result

    # -- internals ----------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call .fit() before requesting recommendations.")

    def _neighbour_candidates(
        self,
        query_vec,
        n: int,
        max_per_artist: int | None,
        exclude_index: int | None,
        exclude_song_key: str | None,
    ) -> list[tuple[int, float]]:
        """Query the NN index, then filter for duplicates and diversity."""
        pool = min(len(self.catalogue), max(config.NEIGHBOUR_POOL_SIZE, n * 10))
        distances, indices = self._nn.kneighbors(query_vec, n_neighbors=pool)

        selected: list[tuple[int, float]] = []
        seen_songs: set[str] = set()
        artist_counts: dict[str, int] = {}
        for dist, i in zip(distances[0], indices[0]):
            i = int(i)
            if exclude_index is not None and i == exclude_index:
                continue
            song_key = self._song_keys[i]
            # Skip other releases of the query song and already-picked songs.
            if song_key == exclude_song_key or song_key in seen_songs:
                continue
            artist_key = self._artist_keys[i]
            if max_per_artist and artist_counts.get(artist_key, 0) >= max_per_artist:
                continue
            selected.append((i, 1.0 - float(dist)))
            seen_songs.add(song_key)
            artist_counts[artist_key] = artist_counts.get(artist_key, 0) + 1
            if len(selected) == n:
                break
        return selected

    def _build_result(
        self,
        pairs: list[tuple[int, float]],
        ref_scaled: np.ndarray,
        ref_raw: pd.Series,
        ref_genres: set[str],
    ) -> pd.DataFrame:
        rows = []
        for rank, (i, similarity) in enumerate(pairs, start=1):
            row = self.catalogue.loc[i]
            rows.append({
                "rank": rank,
                "track_id": row["track_id"],
                "track_name": row["track_name"],
                "artists": row["artists"],
                "album_name": row["album_name"],
                "genres": row["genres"],
                "popularity": row["popularity"],
                "duration": format_duration(row["duration_ms"]),
                "explicit": row["explicit"],
                "similarity_score": round(similarity, 4),
                "why_similar": self._explain(i, ref_scaled, ref_raw, ref_genres),
                "method": "content_based",
            })
        return pd.DataFrame(rows, columns=RESULT_COLUMNS)

    def _explain(
        self,
        cand_idx: int,
        ref_scaled: np.ndarray,
        ref_raw: pd.Series,
        ref_genres: set[str],
    ) -> str:
        """Compose a short, human-readable similarity explanation.

        The features cited are those with the smallest standardised gap
        between query and candidate - i.e. the dimensions on which the two
        tracks genuinely agree most.
        """
        cand_raw = self.catalogue.loc[cand_idx, self.audio_features]
        gaps = np.abs(ref_scaled - self.audio_scaled[cand_idx])
        gap_by_feature = dict(zip(self.audio_features, gaps))
        # Sort by gap, but break near-ties in favour of features where the
        # query track is far from the dataset average. This avoids citing
        # bland agreements (e.g. two pop songs both having ~0 instrumentalness)
        # and instead surfaces the distinctive traits the tracks share.
        closest = sorted(
            self._explain_features,
            key=lambda f: (
                round(gap_by_feature[f], 1),
                -abs(float(ref_scaled[self._feature_pos[f]])),
            ),
        )
        closest = closest[: config.EXPLANATION_FEATURES]

        parts = [
            f"{config.FEATURE_LABELS[f]} "
            f"({format_feature_value(f, float(ref_raw[f]))} vs "
            f"{format_feature_value(f, float(cand_raw[f]))})"
            for f in closest
        ]
        shared = ref_genres & self._genre_sets[cand_idx]
        if shared:
            genre_note = f"shares the '{sorted(shared)[0]}' genre"
        else:
            genre_note = "different genre, but a matching sound profile"
        return "Similar " + ", ".join(parts) + "; " + genre_note
