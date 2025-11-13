# evaluate_recommender.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from typing import List, Optional, Dict, Any
import math
import random

def _ensure_array(x):
    """Convert matrices like np.matrix to ndarray and leave sparse matrices as-is."""
    try:
        # If it's a scipy sparse matrix, return as-is
        from scipy.sparse import issparse
        if issparse(x):
            return x
    except Exception:
        pass
    # else convert to ndarray
    return np.asarray(x)

def _top_k_indices_for_all(tfidf_matrix, k: int):
    """Return top-k indices for each row of tfidf_matrix (excluding itself)."""
    # linear_kernel can accept sparse or dense
    sims = linear_kernel(tfidf_matrix, tfidf_matrix)  # shape (n_docs, n_docs)
    n = sims.shape[0]
    # set diagonal to -inf so self not selected
    np.fill_diagonal(sims, -np.inf)
    # argsort descending
    topk = np.argsort(sims, axis=1)[:, ::-1][:, :k]  # shape (n, k)
    return topk, sims

def evaluate_genre_overlap(
    df: pd.DataFrame,
    tfidf_matrix,
    ks: List[int] = [5, 10, 20],
    sample_frac: Optional[float] = None,
    random_state: int = 42,
    field: str = 'genres_list'
) -> pd.DataFrame:
    """
    Evaluate recommender by genre overlap.

    For each movie i:
      - get top-k recommended indices
      - compute num_relevant_in_topk = number of recommended movies that share at least one
        genre (or keyword/director depending on field) with movie i
      - precision@k = num_relevant_in_topk / k
      - recall@k = num_relevant_in_topk / (total_relevant_in_corpus) where
          total_relevant_in_corpus = number of movies (excluding i) that share >=1 genre with i.
      - f1@k computed from precision & recall safely.

    Returns a DataFrame with rows for each k and aggregated mean metrics.
    """
    if field not in df.columns:
        raise ValueError(f"Field '{field}' not found in DataFrame columns")

    n = len(df)
    indices = list(range(n))
    if sample_frac is not None and 0 < sample_frac < 1.0:
        random.seed(random_state)
        sample_n = max(1, int(n * sample_frac))
        indices = random.sample(indices, sample_n)

    tfidf_matrix = _ensure_array(tfidf_matrix)  # convert if needed

    results = []
    # compute full similarity matrix once (may be large)
    # but it's convenient for top-k and scoring. If memory is an issue, we can compute per-row.
    topk_dict = {}
    sims_matrix = None
    # We'll compute per-row top-k for each requested k (up to max_k)
    max_k = max(ks)
    topk, sims_matrix = _top_k_indices_for_all(tfidf_matrix, max_k)  # topk shape (n, max_k)

    # Precompute each movie's ground-truth set (e.g., genre set)
    truth_sets = []
    for i, row in df.iterrows():
        vals = row.get(field)
        if isinstance(vals, list):
            truth_sets.append(set([str(v) for v in vals if v is not None]))
        else:
            truth_sets.append(set())

    for k in ks:
        precisions = []
        recalls = []
        hits = []  # whether at leas
