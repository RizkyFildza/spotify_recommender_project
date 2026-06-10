# Building a Content-Based Spotify Track Recommender

**Rizky Fildza**

## 1. Introduction

This project builds a recommendation system for music tracks using the Spotify
Tracks Dataset published on Kaggle by Maharshi Pandya. The goal is practical:
given a track the user already likes (or, failing that, a genre and a few
stated preferences), return a ranked list of tracks that *sound* similar, and
explain each recommendation in plain language.

An important design constraint shaped everything that follows: the dataset
contains no user listening histories. There are no play counts per user, no
ratings, and no session logs, so collaborative filtering (the "users who liked
X also liked Y" family of methods) is simply not possible here. The system is
therefore a **content-based recommender**: tracks are compared on what they
are, namely their audio characteristics and genre labels, rather than on who
listens to them. I treat this as a feature of the project rather than a flaw
to hide: it forces the model to be explainable, and it makes the evaluation
question ("how do you measure quality without ground-truth preferences?")
genuinely interesting.

## 2. Dataset overview

The raw file contains 114,000 rows and 21 columns. Each row is a
*(track, genre)* pair carrying the track's Spotify ID, names (track, artists,
album), a popularity score from 0 to 100, and a set of audio features
estimated by Spotify's analysis models: danceability, energy, loudness,
speechiness, acousticness, instrumentalness, liveness, valence, tempo,
duration, key, mode, and time signature.

I audited the file before trusting the Kaggle description, and the audit
surfaced several differences worth recording. The dataset card mentions 125
genres, but the file actually contains **114**, each contributing exactly
1,000 rows. The card documents `time_signature` as ranging from 3 to 7, yet
values 0 and 1 also occur (163 and 973 rows respectively). One row (a k-pop
track) is missing its artist, album, and track name entirely. And 157 rows
report a tempo of 0 BPM while one row reports a duration of 0 ms, values
that are not physically meaningful for real recordings. Popularity behaves as
documented (range 0 to 100, mean 33.2, median 35), though it is worth noting
that 10.4% of unique tracks have a popularity of exactly 0.

The most consequential finding concerns duplication. There are no fully
duplicated rows, but only 89,741 of the 114,000 rows have a unique
`track_id`; the remaining 24,259 rows repeat an id that appears elsewhere. I
checked what actually differs within these repeated ids: **only the genre
label and, occasionally, the popularity score; the audio features are
identical in every case.** In other words, the duplication is an artefact of
how the dataset was assembled (1,000 rows sampled per genre, with the same
track legitimately belonging to several genres), not an inconsistency in the
measurements. This directly informed the cleaning strategy below.

## 3. Data cleaning decisions

The cleaning step is deliberately small, because the data is mostly sound.
I drop the `Unnamed: 0` column (a redundant export index), the single row
with missing metadata (it cannot be displayed or searched, so it cannot be
recommended), and the 157 rows with non-physical tempo or duration values.
Together these removals amount to 158 rows, about 0.14% of the data, leaving
113,842 rows.

The repeated `track_id`s required an actual decision rather than a reflex.
Simply dropping duplicates would silently discard genre information (a track
labelled both *rock* and *grunge* would keep only whichever label happened to
come first), while keeping all rows would let the same recording appear
several times in one recommendation list. Since the audit showed the audio
features are identical within an id, I collapse the data into a **catalogue
of 89,583 unique tracks**, where each track stores the *set* of all genre
labels it appeared under (16,299 tracks end up with more than one) and the
maximum of its popularity values. This keeps every piece of information that
actually varies, removes the redundancy, and gives the recommender a richer,
multi-label view of genre than any single row had.

One further wrinkle: some songs appear under several *different* track ids
(e.g. an album version and a single release with separate ids but the same
name, artist, and essentially the same audio). These cannot be merged safely
in the catalogue, so instead the recommender de-duplicates at query time
using a normalised (track name, primary artist) key, ensuring the same song
never appears twice in one result list.

## 4. Methodology

The similarity space is built from ten numeric audio features:
danceability, energy, loudness, speechiness, acousticness, instrumentalness,
liveness, valence, tempo, and duration. These are standardised with
z-scoring, which is necessary because their raw scales differ enormously
(duration is measured in hundreds of thousands of milliseconds while
danceability lives in [0, 1]); without standardisation, similarity would be
dominated by whichever feature happens to have the largest numbers.

After standardisation I apply per-feature weights, each with a written
rationale in `src/config.py`. Most features keep weight 1.0. Liveness is
down-weighted to 0.5 because audience noise says little about whether two
songs are a good musical match; loudness is down-weighted to 0.7 because the
correlation analysis in the EDA shows it is strongly correlated with energy
(r ≈ 0.76), and giving both full weight would double-count intensity;
duration gets 0.4 because track length matters less than how a track sounds;
tempo gets 0.8 as a mild compromise between rhythmic relevance and its noisy
estimation. I considered including popularity as a similarity feature and
decided against it: popularity measures how often a track is played, not
what it sounds like, and including it would quietly bias the system towards
recommending "similarly famous" tracks. Popularity is still shown in the
output and used by the baseline; it is just not part of *similarity*.

Genre enters as a secondary signal. Each track's genre set becomes a
row-normalised multi-hot vector over the 114 genres, scaled by a weight of
2.0. Since the audio block of ten z-scored features has a typical vector
norm of roughly √10 ≈ 3.2, this makes genre influential, reliably pulling
same-genre tracks up the ranking, without letting it override the audio
evidence. Categorical features I examined and left out are key, mode, and
time signature: musical key relationships are circular (C is as close to B
as to C♯) and a naive encoding would mislead the distance metric, while mode
and time signature are nearly constant across this catalogue and add little.

Similarity itself is cosine distance over the combined 124-dimensional
vectors, computed with scikit-learn's exact brute-force nearest-neighbour
search. At this scale (89,583 tracks) the index builds in about 0.2 seconds
and each query answers in milliseconds, so approximate methods would be
unjustified complexity. The same reasoning ruled out deep learning: there is
no interaction data to learn embeddings from, and a learned model would
sacrifice the explainability that is one of this project's main goals.

## 5. Recommendation system design

Two recommenders are implemented. The **popularity baseline** returns the
most popular tracks either globally or within a chosen genre. It represents
what a system with no notion of similarity would do, and it is the honest
comparison point: any content-based model must demonstrably beat it.

The **content-based recommender** supports three query modes: a track name;
a track name plus artist (which disambiguates covers and common titles;
search is case-insensitive, falls back to substring matching, and resolves
ties by popularity); and a genre/preference profile for users with no seed
track, built from the centroid of the chosen genres with optional overrides
such as `acousticness=0.85, energy=0.25`. For each query the system retrieves
a pool of 300 nearest neighbours and then filters it: the query track itself
is removed, same-song variants are collapsed, and a diversity cap allows at
most two tracks per primary artist, preventing the degenerate outcome where
a Metallica query returns ten Metallica songs.

Every recommendation is explained. The explanation cites the three audio
features on which the query and the candidate agree most closely, measured
as the smallest gap in standardised units, with near-ties broken in favour
of the query's most distinctive traits, so the text highlights meaningful
agreement rather than two pop songs both trivially having zero
instrumentalness. Values are reported in natural units (tempo in BPM,
loudness in dB, duration in minutes and seconds), and a final clause states
the genre relationship. A typical output line reads: *"Similar energy (0.91
vs 0.92), acousticness (0.00 vs 0.00), loudness (-4.6 dB vs -4.6 dB); shares
the 'alt-rock' genre."*

## 6. Evaluation approach

Without user interaction data there is no ground truth for "the user would
have liked this", so accuracy metrics such as precision@k are impossible by
construction. Instead I evaluate with **proxy metrics** over 200 randomly
sampled query tracks (fixed seed for reproducibility), comparing three
systems at top-10: a random recommender, the global popularity baseline, and
the content-based model. Genre overlap@10 measures the fraction of
recommendations sharing at least one genre with the query; audio distance@10
measures the mean Euclidean distance in the *unweighted* standardised audio
space (deliberately not the model's own weighted space, to avoid grading the
model with its own ruler); artist diversity@10 is the share of distinct
primary artists in a list; and mean popularity@10 is reported descriptively
to expose each system's popularity bias.

## 7. Key findings

| system | genre overlap@10 | audio distance@10 | artist diversity@10 | mean popularity@10 |
|---|---|---|---|---|
| random | 0.014 | 4.114 | 0.999 | 32.8 |
| popularity (global) | 0.016 | 3.556 | 0.800 | 97.2 |
| content-based | 0.993 | 1.365 | 0.856 | 33.8 |

The content-based recommender shares at least one genre with the query in
over 99% of cases (genre overlap 0.993 versus 0.016 for the popularity
baseline) and reduces the audio-feature distance to about a third of the
random level (1.365 versus 4.114). In other words, it recommends tracks that
genuinely sound like the query, not merely tracks that are famous. The
popularity baseline's mean recommendation popularity of 97.2/100 makes its
failure mode explicit: it returns the same handful of global hits regardless
of what was asked. The content-based model's mean popularity of 33.8 sits at
the catalogue average, confirming that excluding popularity from the
similarity space removed that bias. The diversity cap holds artist diversity
at 0.856, slightly below the random recommender's near-perfect 0.999, which
is expected and acceptable, since an artist's own songs are often
legitimately the most similar tracks.

Qualitative spot checks support the numbers. *Master of Puppets* returns
Megadeth, Guns N' Roses, Iron Maiden, and Dio; a jazz profile with high
acousticness and low energy returns Eartha Kitt, Dean Martin, Nat King Cole,
and Louis Armstrong; *Shape of You* surfaces dance-pop with similar rhythmic
profiles. The exploratory analysis also validates the feature space itself:
the genre audio-profile figure shows the features cleanly separate
contrasting genres (classical: acousticness 0.92, energy 0.19; death-metal:
acousticness 0.01, energy 0.93), which is precisely the property a
content-based recommender depends on.

## 8. Limitations

These results should be read with several honest caveats. First and most
fundamentally, the proxy metrics measure *similarity*, not *satisfaction*;
without user data I cannot verify that similar-sounding tracks are what a
listener actually wants, and a production system would need A/B testing or
historical interaction data to close that gap. Second, the catalogue is a
static snapshot from 2022, so the system cannot recommend newer releases and
suffers a cold-start problem for any query track outside the catalogue.
Third, the genre labels are noisy: each row carried a single label from one
source, so the multi-genre sets recovered by de-duplication are incomplete
rather than authoritative, and the near-perfect genre-overlap score partly
reflects that genre is itself an input to the model. Fourth, the audio
features are outputs of Spotify's proprietary estimation models, not ground
truth measurements, and inherit whatever biases those models carry. Finally,
the explanations are heuristic summaries of feature proximity; they make the
ranking transparent, but they are not causal claims about why a listener
would enjoy a track.

## 9. Conclusion

The project delivers what it set out to: a clean, modular, reproducible
content-based recommender that demonstrably outperforms a popularity
baseline on genre consistency and audio similarity, avoids popularity bias
by design, returns diverse and de-duplicated lists, and explains every
recommendation it makes. The methodological choices (weighted standardised
features, genre as a secondary signal, exact cosine nearest neighbours, and
proxy evaluation against explicit baselines) are each justified by
properties observed in the data rather than imported by habit. The main
avenue for future work is obtaining interaction data, which would enable
hybrid approaches and genuine preference-based evaluation.

## 10. Acknowledgements

I developed this project with the assistance of Claude Fable 5 (Anthropic),
which I used as a pair programmer and writing aid. The methodology,
data decisions, and conclusions in this report reflect my own
understanding of the work.

## 11. Dataset citation

Maharshi Pandya (2022). *🎹 Spotify Tracks Dataset*. Kaggle.
DOI: [10.34740/KAGGLE/DSV/4372070](https://doi.org/10.34740/kaggle/dsv/4372070)

```bibtex
@misc{maharshi_pandya_2022,
  title     = {🎹 Spotify Tracks Dataset},
  url       = {https://www.kaggle.com/dsv/4372070},
  DOI       = {10.34740/KAGGLE/DSV/4372070},
  publisher = {Kaggle},
  author    = {Maharshi Pandya},
  year      = {2022}
}
```
