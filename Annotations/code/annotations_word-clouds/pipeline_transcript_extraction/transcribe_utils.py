"""Utility functions for ZNLRESET transcript NLP pipeline.

Parses Whisper chunk files of the form

    [VIDEO TIMESTAMP: HH:MM:SS.fff][QUALITY WARNINGS: silence_detected]Body text...

and provides English NLP preprocessing analogous to the original
``BreathWork_Utils`` from the RESET_Annotation reference repo, with dynamic
subplot grids instead of the hardcoded 4x4 layout.
"""

from __future__ import annotations

import difflib
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chunk parsing
# ---------------------------------------------------------------------------

# Match `[VIDEO TIMESTAMP: HH:MM:SS.fff]` followed optionally by
# `[QUALITY WARNINGS: csv,of,warnings]`, then the body. DOTALL so multi-line
# bodies (newlines inside the transcript) are captured. Whitespace (incl.
# newlines) between brackets is tolerated — Whisper sometimes inserts a space
# or line break between the timestamp and the warnings tag.
CHUNK_HEADER_RE = re.compile(
    r"\[VIDEO TIMESTAMP:\s*(?P<ts>\d{2}:\d{2}:\d{2}\.\d{3})\]"
    r"\s*"
    r"(?:\[QUALITY WARNINGS:\s*(?P<warn>[^\]]*)\]\s*)?"
    r"(?P<body>.*)",
    re.DOTALL,
)

# Manually-cleaned chunks carry a coding tag on the line directly below the
# header, e.g. `[experience_body]`. Matching is fuzzy (see _match_filter_tag)
# so a typo like `[experiance_body]` is still recognized.
FILTER_TAGS: tuple[str, ...] = (
    "experience",
    "experience_affect",
    "experience_body",
    "experience_insight",
)

# A first line that is *only* a bracketed token — nothing else on the line.
_FILTER_TAG_LINE_RE = re.compile(r"^\[\s*(.+?)\s*\]\s*$")

_FILTER_TAG_FUZZY_CUTOFF = 0.72


@dataclass(frozen=True)
class ParsedChunk:
    """Result of parsing a single Whisper-chunk ``.txt`` file."""

    path: Path
    timestamp_s: float
    timestamp_str: str
    quality_warnings: tuple[str, ...]
    text: str
    filter_tag: str | None = None


def hms_to_seconds(hms: str) -> float:
    """Convert ``HH:MM:SS.fff`` to seconds (float).

    Parameters
    ----------
    hms : str
        Timestamp string with millisecond precision.

    Returns
    -------
    float
        Total seconds.
    """
    h, m, rest = hms.split(":")
    s = float(rest)
    return int(h) * 3600 + int(m) * 60 + s


def _match_filter_tag(candidate: str, tags: tuple[str, ...] = FILTER_TAGS,
                      cutoff: float = _FILTER_TAG_FUZZY_CUTOFF) -> str | None:
    """Fuzzy-match ``candidate`` against the canonical ``tags``.

    Exact match (after lowercasing / space-to-underscore normalization) wins
    outright; otherwise falls back to ``difflib`` similarity so a typo like
    ``experiance_body`` still resolves to ``experience_body``.
    """
    normalized = candidate.strip().lower().replace(" ", "_")
    if normalized in tags:
        return normalized
    matches = difflib.get_close_matches(normalized, tags, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def extract_filter_tag(body: str) -> tuple[str | None, str | None, str]:
    """Strip a coding tag from the first line of ``body``, if present.

    Parameters
    ----------
    body : str
        Chunk body text (header already removed), leading/trailing
        whitespace already stripped by the caller.

    Returns
    -------
    tuple
        ``(filter_tag, unrecognized_raw, remaining_text)`` — exactly one of
        ``filter_tag`` / ``unrecognized_raw`` is non-``None`` when the first
        line was a bracket-only token; both are ``None`` when the first line
        was ordinary transcript text (``remaining_text`` then equals
        ``body`` unchanged). A bracket-only first line is always removed
        from the returned text, matched or not.
    """
    newline_idx = body.find("\n")
    first_line = body if newline_idx == -1 else body[:newline_idx]
    m = _FILTER_TAG_LINE_RE.match(first_line.strip())
    if m is None:
        return None, None, body

    raw = m.group(1).strip()
    remainder = "" if newline_idx == -1 else body[newline_idx + 1:].strip()
    tag = _match_filter_tag(raw)
    if tag is not None:
        return tag, None, remainder
    return None, raw, remainder


def parse_chunk_file(path: Path | str) -> ParsedChunk:
    """Parse a single Whisper chunk file.

    Parameters
    ----------
    path : Path or str
        Path to a ``.txt`` file produced by the upstream Whisper pipeline.

    Returns
    -------
    ParsedChunk
        Structured representation of the chunk.

    Raises
    ------
    ValueError
        If the file content does not match the expected
        ``[VIDEO TIMESTAMP: ...]...`` header pattern.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    m = CHUNK_HEADER_RE.match(raw)
    if m is None:
        raise ValueError(f"Chunk file does not match expected header pattern: {path}")

    ts_str = m.group("ts")
    warn_csv = m.group("warn") or ""
    warnings_tuple = tuple(w.strip() for w in warn_csv.split(",") if w.strip())
    body = m.group("body").strip()

    filter_tag, unrecognized_raw, body = extract_filter_tag(body)
    if unrecognized_raw is not None:
        logger.warning(
            "Unrecognized filter tag %r in %s — stripped from text, logged as quality warning",
            unrecognized_raw, path,
        )
        warnings_tuple = warnings_tuple + (f"unrecognized_filter_tag:{unrecognized_raw}",)

    return ParsedChunk(
        path=path,
        timestamp_s=hms_to_seconds(ts_str),
        timestamp_str=ts_str,
        quality_warnings=warnings_tuple,
        text=body,
        filter_tag=filter_tag,
    )


# ---------------------------------------------------------------------------
# Text preprocessing (port of BreathWork_Utils.preprocess)
# ---------------------------------------------------------------------------


def _get_wordnet_pos(tag: str):
    """Map NLTK POS tag to WordNet POS for lemmatization."""
    from nltk.corpus import wordnet

    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_words(words: list[list[str]]) -> list[list[str]]:
    """Lemmatize a list of tokenized sentences.

    Parameters
    ----------
    words : list of list of str
        Sentences, each as a list of word tokens.

    Returns
    -------
    list of list of str
        Lemmatized sentences with the same outer structure.
    """
    import nltk
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    lemmatized_sentences: list[list[str]] = []
    for sent_words in words:
        lemmatized_words: list[str] = []
        for word in sent_words:
            pos = _get_wordnet_pos(nltk.pos_tag([word])[0][1])
            lemmatized_words.append(lemmatizer.lemmatize(word, pos))
        lemmatized_sentences.append(lemmatized_words)
    return lemmatized_sentences


def flatten(items: Iterable):
    """Recursively flatten arbitrarily nested iterables of strings."""
    for it in items:
        if isinstance(it, list):
            yield from flatten(it)
        else:
            yield it


def preprocess(texts: Iterable[str]) -> pd.DataFrame:
    """Tokenize, lowercase, strip punctuation and remove English stopwords.

    Mirrors ``BreathWork_Utils.preprocess`` from the reference repo so that
    EDA outputs are directly comparable.

    Parameters
    ----------
    texts : iterable of str
        Raw transcript bodies (one per "document" — typically a chunk or
        a concatenation of chunks at the chosen aggregation level).

    Returns
    -------
    pandas.DataFrame
        Columns ``text``, ``sentences``, ``words`` (a list-of-list-of-tokens).
    """
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize

    stop_words = set(stopwords.words("english"))
    rows: list[dict] = []
    for text in texts:
        cleaned = text.lower().replace("\n", " ")
        sentences = sent_tokenize(cleaned)
        words = [word_tokenize(sent) for sent in sentences]
        words = [[w.lower() for w in sent_words if w.isalnum()] for sent_words in words]
        words = [[w for w in sent_words if w not in stop_words] for sent_words in words]
        rows.append({"text": cleaned, "sentences": sentences, "words": words})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization helpers (dynamic grids instead of hardcoded 4x4)
# ---------------------------------------------------------------------------


def _grid_dims(n: int) -> tuple[int, int]:
    """Return (nrows, ncols) for a near-square grid that fits ``n`` panels."""
    if n <= 0:
        raise ValueError("n must be positive")
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def basic_statistics(df: pd.DataFrame) -> dict[str, float]:
    """Print and return basic corpus statistics from a preprocessed dataframe.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`preprocess`.

    Returns
    -------
    dict
        Statistics keyed by name.
    """
    num_documents = len(df)
    total_sentences = int(df["sentences"].apply(len).sum())
    total_words = int(
        df["words"].apply(lambda x: sum(len(s) for s in x)).sum()
    )
    unique_words = len(
        {w for doc in df["words"] for sent in doc for w in sent}
    )
    stats = {
        "num_documents": num_documents,
        "total_sentences": total_sentences,
        "avg_sentences_per_document": total_sentences / max(num_documents, 1),
        "total_words": total_words,
        "avg_words_per_sentence": total_words / max(total_sentences, 1),
        "unique_words": unique_words,
    }
    for k, v in stats.items():
        print(f"{k}: {v}")
    return stats


def visualize_length_distribution(df: pd.DataFrame) -> None:
    """Plot sentence- and word-length violin distributions."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sentence_lengths = df["sentences"].apply(len)
    word_lengths = df["words"].apply(lambda x: sum(len(s) for s in x))

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.violinplot(data=sentence_lengths)
    plt.title("Sentence Length Distribution")
    plt.xlabel("")

    plt.subplot(1, 2, 2)
    sns.violinplot(data=word_lengths)
    plt.title("Word Length Distribution")
    plt.xlabel("")

    sns.despine()
    plt.tight_layout()


def generate_word_cloud(
    df: pd.DataFrame,
    words_col: str = "words",
    mask_path: Path | str | None = None,
    max_words: int = 100,
):
    """Render a global WordCloud over all documents.

    Parameters
    ----------
    df : DataFrame
        Output of :func:`preprocess` (or a frame with a comparable list-col).
    words_col : str
        Column holding nested word lists (default ``"words"``).
    mask_path : Path or str, optional
        PNG mask for cloud shape. ``None`` → rectangular cloud.
    max_words : int
        Max words rendered in the cloud.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    from wordcloud import WordCloud

    all_words = " ".join(flatten(df[words_col].tolist()))
    mask = np.array(Image.open(mask_path)) if mask_path is not None else None
    wc = WordCloud(
        mask=mask,
        background_color="white",
        max_words=max_words,
        collocations=True,
        contour_color="black",
        contour_width=1 if mask is not None else 0,
    ).generate(all_words)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    sns.despine()
    return wc


def plot_top_bigrams(
    df: pd.DataFrame,
    words_col: str = "words",
    top_n: int = 5,
    subject_col: str | None = None,
) -> None:
    """Plot top-N bigrams per row of ``df`` in a dynamic grid.

    Parameters
    ----------
    df : DataFrame
        One row = one subject / aggregation unit.
    words_col : str
        Column holding nested word lists.
    top_n : int
        Number of bigrams per subject.
    subject_col : str, optional
        Column used for subplot titles. Falls back to row index.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

    n = len(df)
    nrows, ncols = _grid_dims(n)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        bigram_measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(flatten(row[words_col]))
        finder.apply_freq_filter(1)
        top_bigrams = finder.nbest(bigram_measures.raw_freq, top_n)
        freqs = [finder.ngram_fd[bg] for bg in top_bigrams]
        labels = [f"{a} {b}" for (a, b) in top_bigrams]
        if labels:
            sns.barplot(x=labels, y=freqs, ax=ax, color="gray")
            ax.set_xticklabels(labels, rotation=45, ha="right")
        title = row[subject_col] if subject_col and subject_col in df.columns else f"Subject {i+1}"
        ax.set_title(str(title))
        ax.set_ylabel("Frequency")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    sns.despine()


def ensure_nltk_resources() -> None:
    """Download the NLTK corpora required for preprocessing if missing."""
    import nltk

    for resource in (
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
        "corpora/stopwords",
        "corpora/wordnet",
        "taggers/averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng",
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource.split("/", 1)[1])



