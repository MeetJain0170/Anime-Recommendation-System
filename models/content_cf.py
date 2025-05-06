import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

class ContentBasedFiltering:
    def __init__(self, anime_df):
        self.anime_df = anime_df.copy()
        self.feature_matrix = None
        self.similarity_matrix = None
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))  # using bigrams

    def build_feature_matrix(self):
        # Genre columns (one-hot encoded)
        genre_columns = ['Action', 'Adventure', 'Cars', 'Comedy', 'Dementia', 'Demons', 'Drama', 'Ecchi',
                         'Fantasy', 'Game', 'Harem', 'Hentai', 'Historical', 'Horror', 'Josei', 'Kids',
                         'Magic', 'Martial Arts', 'Mecha', 'Military', 'Music', 'Mystery', 'Parody', 'Police',
                         'Psychological', 'Romance', 'Samurai', 'School', 'Sci-Fi', 'Seinen', 'Shoujo',
                         'Shoujo Ai', 'Shounen', 'Shounen Ai', 'Slice of Life', 'Space', 'Sports', 'Super Power',
                         'Supernatural', 'Thriller', 'Vampire']

        # Reconstruct 'genre' as a string list from one-hot columns
        self.anime_df['genre'] = self.anime_df[genre_columns].apply(
            lambda row: ' '.join([col for col in genre_columns if row[col] == 1]),
            axis=1
        )

        # Combine all content features into a single string
        self.anime_df['content_features'] = (
            self.anime_df['combined_text'].fillna('') + ' ' +
            self.anime_df['genre'].fillna('') + ' ' +
            self.anime_df['type'].fillna('') + ' ' +
            self.anime_df['source'].fillna('') + ' ' +
            self.anime_df['studio'].fillna('')
        )

        # Apply TF-IDF vectorization with bigrams and trigrams for better feature extraction
        tfidf_matrix = self.vectorizer.fit_transform(self.anime_df['content_features'])

        # Normalize numerical features
        num_features = self.anime_df[['normalized_score', 'aired_from_year', 'duration_min']].fillna(0)
        scaler = MinMaxScaler()
        scaled_num_features = scaler.fit_transform(num_features)

        # Add additional features: average rating and popularity
        self.anime_df['average_rating'] = self.anime_df['normalized_score'].mean()  # replace with actual ratings if available
        self.anime_df['popularity'] = self.anime_df['aired_from_year']  # replace with actual popularity data if available

        # Concatenate text and numeric features
        self.feature_matrix = hstack([tfidf_matrix, scaled_num_features])

    def compute_similarity(self):
        if self.feature_matrix is None:
            self.build_feature_matrix()
        # Use cosine similarity for better feature-based similarity
        self.similarity_matrix = cosine_similarity(self.feature_matrix)

    def get_similarity_score(self, username, target_anime_id, ratings_df):
        """
        Compute weighted similarity score of a target anime against all animes the user has rated.
        The weights are the user's rating scores (normalized 0–1).
        """
        if self.similarity_matrix is None:
            self.compute_similarity()

        # Get index of the target anime
        try:
            target_idx = self.anime_df[self.anime_df['anime_id'] == target_anime_id].index[0]
        except IndexError:
            return 0.0

        # Get all animes the user has rated
        user_ratings = ratings_df[ratings_df['username'] == username]
        if user_ratings.empty:
            return 0.0

        score = 0.0
        weight_sum = 0.0

        for _, row in user_ratings.iterrows():
            try:
                rated_anime_id = row['anime_id']
                rating = row['rating'] / 10.0  # normalize

                rated_idx = self.anime_df[self.anime_df['anime_id'] == rated_anime_id].index[0]
                sim = self.similarity_matrix[target_idx][rated_idx]

                score += sim * rating
                weight_sum += rating
            except:
                continue

        if weight_sum == 0:
            return 0.0

        return score / weight_sum

    
    def recommend(self, username, ratings_df, candidate_anime_ids, top_n=10):
        """
        Recommend top-N anime for a user from a list of candidate_anime_ids based on content similarity.
        Returns a list of (anime_id, score) tuples.
        """
        scored_candidates = []
        for anime_id in candidate_anime_ids:
            try:
                score = self.get_similarity_score(username, anime_id, ratings_df)
            except:
                score = 0.0  # fallback
            scored_candidates.append((anime_id, score))

        # Sort by score descending
        top_recommendations = sorted(scored_candidates, key=lambda x: x[1], reverse=True)[:top_n]
        return top_recommendations


