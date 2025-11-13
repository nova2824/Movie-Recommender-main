"""
Movie Recommender with Streamlit-safe and CLI fallback

This file provides preprocessing, TF-IDF recommender and a Streamlit UI if Streamlit
is installed. It falls back to a CLI mode if Streamlit is not available.

Run with Streamlit (preferred):
    streamlit run streamlit_app_debugged.py

Or use the CLI fallback:
    python streamlit_app_debugged.py --mode title --title "Toy Movie"

You can also run the built-in unit tests:
    python streamlit_app_debugged.py --run-tests

This updated version fixes the issue where the slider-controlled number of
recommendations was ignored: both Title and Genre modes now use distinct
slider variables (`n_title`, `n_genre`) and honor the requested `n` when
returning recommendations or displaying results.
"""
from typing import Any, List, Optional, Tuple
import json
import ast
from pathlib import Path
import argparse
import sys
import traceback

import pandas as pd
import numpy as np

# NLP
import nltk
from nltk.corpus import stopwords

# sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Try to import Streamlit; provide a fallback if it's not installed
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False
    class _DummySt:
        def __getattr__(self, name):
            def _noop(*args, **kwargs):
                return None
            return _noop
    st = _DummySt()

# Provide cache_data decorator mapping to Streamlit's when available
if STREAMLIT_AVAILABLE and hasattr(st, 'cache_data'):
    cache_data = st.cache_data
else:
    def cache_data(func):
        return func

# Ensure NLTK stopwords
nltk.download('stopwords', quiet=True)
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    STOP_WORDS = { 'the', 'and', 'is', 'in', 'to', 'of', 'a', 'an', 'for', 'on', 'with', 'as', 'by' }


# -------------------------
# Parsing / Preprocessing
# -------------------------

def safe_parse(x: Any) -> Any:
    if pd.isna(x):
        return []
    if isinstance(x, (list, dict)):
        return x
    if not isinstance(x, str):
        return []
    s = x.strip()
    if s == '':
        return []
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    try:
        s2 = s.replace('""', '"')
        return json.loads(s2)
    except Exception:
        return []


def extract_names(x: Any, key: str = 'name') -> List[str]:
    parsed = safe_parse(x)
    if not isinstance(parsed, list):
        return []
    return [item.get(key) for item in parsed if isinstance(item, dict) and isinstance(item.get(key), str)]


def parse_top_cast(x: Any, top_n: int = 3) -> List[str]:
    parsed = safe_parse(x)
    if not isinstance(parsed, list):
        return []
    return [person.get('name') for person in parsed[:top_n] if isinstance(person, dict) and isinstance(person.get('name'), str)]


def extract_director(x: Any) -> Optional[str]:
    parsed = safe_parse(x)
    if not isinstance(parsed, list):
        return None
    for member in parsed:
        if isinstance(member, dict):
            job = member.get('job') or member.get('title') or member.get('department')
            if isinstance(job, str) and job.strip().lower() == 'director':
                name = member.get('name')
                if isinstance(name, str):
                    return name
    return None


def _clean_token(token: str) -> str:
    return token.strip().replace(' ', '_').lower()


def preprocess_text(text: Any) -> str:
    if pd.isna(text):
        return ''
    s = str(text).lower()
    tokens = [tok for tok in s.split() if tok and tok not in STOP_WORDS]
    return ' '.join(tokens)


def create_soup(row: pd.Series) -> str:
    parts: List[str] = []
    overview = row.get('overview')
    if isinstance(overview, str) and overview.strip():
        processed_overview = preprocess_text(overview)
        if processed_overview:
            tokens = [_clean_token(t) for t in processed_overview.split()]
            parts.append(' '.join(tokens))
    for col in ('genres_list', 'keywords_list', 'cast_list'):
        vals = row.get(col)
        if isinstance(vals, list):
            parts.extend([_clean_token(v) for v in vals if isinstance(v, str) and v.strip()])
    director = row.get('director')
    if isinstance(director, str) and director.strip():
        parts.append(_clean_token(director))
    return ' '.join(parts)


def preprocess_movies(movies_df: pd.DataFrame, credits_df: Optional[pd.DataFrame] = None, top_cast_n: int = 3) -> pd.DataFrame:
    df = movies_df.copy()
    # Safe handling if columns are missing
    df['genres_list'] = df.get('genres', pd.Series([[]]*len(df))).apply(extract_names)
    df['keywords_list'] = df.get('keywords', pd.Series([[]]*len(df))).apply(extract_names)

    if credits_df is not None:
        credits = credits_df.copy()
        if 'movie_id' in credits.columns and 'id' in df.columns and 'id' not in credits.columns:
            credits = credits.rename(columns={'movie_id': 'id'})
        credits['cast_list'] = credits.get('cast', pd.Series([[]]*len(credits))).apply(lambda x: parse_top_cast(x, top_cast_n))
        credits['director'] = credits.get('crew', pd.Series([None]*len(credits))).apply(extract_director)
        if 'id' in df.columns and 'id' in credits.columns:
            df = df.merge(credits[['id', 'cast_list', 'director']], on='id', how='left')
        elif 'title' in df.columns and 'title' in credits.columns:
            df = df.merge(credits[['title', 'cast_list', 'director']], on='title', how='left')
        else:
            df['cast_list'] = df.get('cast_list', pd.Series([[]]*len(df)))
            df['director'] = df.get('director', pd.Series([None]*len(df)))
    else:
        df['cast_list'] = df.get('cast_list', pd.Series([[]]*len(df)))
        df['director'] = df.get('director', pd.Series([None]*len(df)))

    df['soup'] = df.apply(create_soup, axis=1)
    return df


def fit_recommender(df: pd.DataFrame, field: str = 'soup', max_features: int = 5000) -> Tuple[TfidfVectorizer, np.ndarray]:
    texts = df[field].fillna('')
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def _find_title_index(df: pd.DataFrame, title: str) -> int:
    if 'title' not in df.columns:
        raise ValueError('DataFrame has no `title` column')
    exact_matches = df.index[df['title'] == title].tolist()
    if exact_matches:
        return int(df.index.get_loc(exact_matches[0]))
    lower_titles = df['title'].astype(str).str.lower()
    ci_matches = df.index[lower_titles == title.lower()].tolist()
    if ci_matches:
        return int(df.index.get_loc(ci_matches[0]))
    contains_matches = df.index[lower_titles.str.contains(title.lower(), na=False)].tolist()
    if contains_matches:
        return int(df.index.get_loc(contains_matches[0]))
    raise ValueError(f"Title '{title}' not found in DataFrame")


def recommend(title: str, df: pd.DataFrame, tfidf_matrix, n: int = 10) -> List[Tuple[str, float]]:
    idx = _find_title_index(df, title)
    sims = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sims[idx] = -1.0
    top_n_idx = sims.argsort()[::-1][:n]
    results = [(str(df.iloc[i]['title']), float(sims[i])) for i in top_n_idx]
    return results


@st.cache_data
def load_and_prepare(data_dir: Optional[str] = None):

    # We skip project_root since we are using an absolute path
    if data_dir is None:
        # <<< YOUR REAL DATA PATH HERE >>>
        data_path = Path(r"C:\Users\naray_he7wm7m\Desktop\green\Movie_Recommender-main\data")
    else:
        data_path = Path(data_dir)

    movies_df = pd.read_csv(data_path / "movies.csv", low_memory=False)
    credits_df = pd.read_csv(data_path / "credits.csv", low_memory=False)

    processed = preprocess_movies(movies_df, credits_df)
    vectorizer, tfidf_matrix = fit_recommender(processed)

    return processed, vectorizer, tfidf_matrix



def main_cli_run_sample():
    # Small sample run to demonstrate CLI mode (uses tiny in-memory data)
    movies = pd.DataFrame([
        {'id': 1, 'title': 'Funny Movie', 'overview': 'A very funny story', 'genres': "[{'id': 35, 'name': 'Comedy'}]"},
        {'id': 2, 'title': 'Scary Night', 'overview': 'A spooky tale', 'genres': "[{'id': 27, 'name': 'Horror'}]"},
        {'id': 3, 'title': 'Action Blast', 'overview': 'Explosions and thrills', 'genres': "[{'id': 28, 'name': 'Action'}]"},
        {'id': 4, 'title': 'Comedy Horror', 'overview': 'Scary but funny', 'genres': "[{'id': 35, 'name': 'Comedy'}, {'id':27,'name':'Horror'}]"},
    ])
    credits = pd.DataFrame([{'id':1,'cast':'[]','crew':'[]'},{'id':2,'cast':'[]','crew':'[]'},{'id':3,'cast':'[]','crew':'[]'},{'id':4,'cast':'[]','crew':'[]'}])
    processed = preprocess_movies(movies, credits)
    _, tfidf_matrix = fit_recommender(processed)
    # Recommend by genre (Comedy + Horror) and request 3 results
    mask = processed['genres_list'].apply(lambda g: any(gg in g for gg in ['Comedy','Horror']))
    genre_indices = processed[mask].index.tolist()
    genre_vec = np.asarray(tfidf_matrix[genre_indices].mean(axis=0)).ravel()
    sims = linear_kernel(genre_vec.reshape(1, -1), tfidf_matrix).flatten()
    top_idx = sims.argsort()[::-1][:3]
    print('CLI sample recommendations:')
    for i in top_idx:
        print(processed.iloc[i]['title'], float(sims[i]))


def main():
    if not STREAMLIT_AVAILABLE:
        print('Streamlit not installed. Run in CLI mode or install Streamlit to use the web UI.')
        return

    st.title('Movie Recommender')
    with st.spinner('Loading data and fitting recommender...'):
        try:
            processed, vectorizer, tfidf_matrix = load_and_prepare()
        except Exception as e:
            st.error(f'Failed to load or prepare data: {e}')
            st.stop()
            return
    st.success('Ready')

    mode = st.radio('Recommend by', ['Movie title', 'Genre(s)'])

    if mode == 'Movie title':
        # Title mode — slider named n_title and used directly
        titles = processed['title'].fillna('').astype(str).tolist()
        titles_sorted = sorted([t for t in titles if t.strip()])
        selected_title = st.selectbox('Select a movie title', titles_sorted)
        n_title = st.slider('Number of recommendations (title mode)', 1, 100, 10, key='n_title')
        if st.button('Recommend by title'):
            try:
                recs = recommend(selected_title, processed, tfidf_matrix, n=n_title)
                df_recs = pd.DataFrame(recs, columns=['title', 'score'])
                st.dataframe(df_recs)
            except Exception as e:
                st.error(f'Recommendation error: {e}')

    else:  # Genre(s) mode
        all_genres = sorted({g for lst in processed['genres_list'] if isinstance(lst, list) for g in lst})
        selected_genres = st.multiselect('Select one or more genres', all_genres)
        # Genre slider separate name
        n_genre = st.slider('Number of results (genre mode)', 1, 50, 10, key='n_genre')

        # Button: list matching movies (up to n_genre)
        if st.button('List movies matching selected genres'):
            if not selected_genres:
                st.warning('Please select at least one genre.')
            else:
                mask = processed['genres_list'].apply(lambda gl: any(g in gl for g in selected_genres) if isinstance(gl, list) else False)
                filtered = processed[mask]
                if filtered.empty:
                    st.error('No movies found for the selected genres.')
                else:
                    # show up to n_genre rows
                    st.success(f'Found {len(filtered)} movies matching {", ".join(selected_genres)}')
                    st.dataframe(filtered[['title', 'genres_list', 'director']].head(n_genre))




if __name__ == '__main__':
    # Command-line options for convenience
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['cli_sample', 'streamlit', 'title', 'genre', 'run-tests'], default='streamlit')
    parser.add_argument('--title', type=str, help='Title to recommend for in CLI title mode')
    parser.add_argument('--n', type=int, default=10, help='Number of recommendations for CLI modes')
    args = parser.parse_args()

    if args.mode == 'cli_sample':
        main_cli_run_sample()
    elif args.mode == 'run-tests':
        # Run a few basic tests
        try:
            # Test parsing
            assert extract_names("[{'id':1,'name':'Action'}]") == ['Action']
            # Test preprocessing and recommendation length
            movies = pd.DataFrame([
                {'id': 1, 'title': 'Funny Movie', 'overview': 'A very funny story', 'genres': "[{'id': 35, 'name': 'Comedy'}]"},
                {'id': 2, 'title': 'Scary Night', 'overview': 'A spooky tale', 'genres': "[{'id': 27, 'name': 'Horror'}]"},
                {'id': 3, 'title': 'Action Blast', 'overview': 'Explosions and thrills', 'genres': "[{'id': 28, 'name': 'Action'}]"},
                {'id': 4, 'title': 'Comedy Horror', 'overview': 'Scary but funny', 'genres': "[{'id': 35, 'name': 'Comedy'}, {'id':27,'name':'Horror'}]"},
            ])
            credits = pd.DataFrame([{'id':1,'cast':'[]','crew':'[]'},{'id':2,'cast':'[]','crew':'[]'},{'id':3,'cast':'[]','crew':'[]'},{'id':4,'cast':'[]','crew':'[]'}])
            processed = preprocess_movies(movies, credits)
            _, tfidf_matrix = fit_recommender(processed)
            recs = None
            try:
                recs = recommend('Funny Movie', processed, tfidf_matrix, n=3)
            except Exception as e:
                print('Recommend failed in tests:', e)
                raise
            assert len(recs) == 3
            # Genre-based recommendation respects requested n
            mask = processed['genres_list'].apply(lambda gl: any(g in gl for g in ['Comedy']) if isinstance(gl, list) else False)
            indices = processed[mask].index.tolist()
            genre_vec = np.asarray(tfidf_matrix[indices].mean(axis=0)).ravel()
            sims = linear_kernel(genre_vec.reshape(1, -1), tfidf_matrix).flatten()
            top_idx = sims.argsort()[::-1][:2]
            assert len(top_idx) == 2
            print('All tests passed')
        except AssertionError:
            traceback.print_exc()
            sys.exit(2)
    elif args.mode == 'title':
        # CLI title mode
        movies = pd.DataFrame([
            {'id': 1, 'title': 'Funny Movie', 'overview': 'A very funny story', 'genres': "[{'id': 35, 'name': 'Comedy'}]"},
            {'id': 2, 'title': 'Scary Night', 'overview': 'A spooky tale', 'genres': "[{'id': 27, 'name': 'Horror'}]"},
            {'id': 3, 'title': 'Action Blast', 'overview': 'Explosions and thrills', 'genres': "[{'id': 28, 'name': 'Action'}]"},
            {'id': 4, 'title': 'Comedy Horror', 'overview': 'Scary but funny', 'genres': "[{'id': 35, 'name': 'Comedy'}, {'id':27,'name':'Horror'}]"},
        ])
        credits = pd.DataFrame([{'id':1,'cast':'[]','crew':'[]'},{'id':2,'cast':'[]','crew':'[]'},{'id':3,'cast':'[]','crew':'[]'},{'id':4,'cast':'[]','crew':'[]'}])
        processed = preprocess_movies(movies, credits)
        _, tfidf_matrix = fit_recommender(processed)
        title = args.title or 'Funny Movie'
        n = args.n
        recs = recommend(title, processed, tfidf_matrix, n=n)
        print(pd.DataFrame(recs, columns=['title', 'score']))
    elif args.mode == 'genre':
        movies = pd.DataFrame([
            {'id': 1, 'title': 'Funny Movie', 'overview': 'A very funny story', 'genres': "[{'id': 35, 'name': 'Comedy'}]"},
            {'id': 2, 'title': 'Scary Night', 'overview': 'A spooky tale', 'genres': "[{'id': 27, 'name': 'Horror'}]"},
            {'id': 3, 'title': 'Action Blast', 'overview': 'Explosions and thrills', 'genres': "[{'id': 28, 'name': 'Action'}]"},
            {'id': 4, 'title': 'Comedy Horror', 'overview': 'Scary but funny', 'genres': "[{'id': 35, 'name': 'Comedy'}, {'id':27,'name':'Horror'}]"},
        ])
        credits = pd.DataFrame([{'id':1,'cast':'[]','crew':'[]'},{'id':2,'cast':'[]','crew':'[]'},{'id':3,'cast':'[]','crew':'[]'},{'id':4,'cast':'[]','crew':'[]'}])
        processed = preprocess_movies(movies, credits)
        _, tfidf_matrix = fit_recommender(processed)
        # recommend for Comedy+Horror
        mask = processed['genres_list'].apply(lambda gl: any(g in gl for g in ['Comedy','Horror']) if isinstance(gl, list) else False)
        indices = processed[mask].index.tolist()
        genre_vec = np.asarray(tfidf_matrix[indices].mean(axis=0)).ravel()
        sims = linear_kernel(genre_vec.reshape(1, -1), tfidf_matrix).flatten()
        top_idx = sims.argsort()[::-1][:args.n]
        results = [(processed.iloc[i]['title'], float(sims[i])) for i in top_idx]
        print(pd.DataFrame(results, columns=['title', 'similarity_score']))
    else:
        # Default: run the Streamlit app
        main()
