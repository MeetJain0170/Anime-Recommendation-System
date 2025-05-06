# models/item_cf.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ItemCF:
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df
        self.item_user_matrix = None
        self.similarity_matrix = None

    def train(self):
        df = self.ratings_df[['username', 'anime_id', 'normalized_rating']].dropna()
        self.item_user_matrix = df.pivot_table(index='anime_id', columns='username', values='normalized_rating').fillna(0)
        self.similarity_matrix = cosine_similarity(self.item_user_matrix)

    def recommend(self, anime_id, top_k=10):
        if self.item_user_matrix is None or self.similarity_matrix is None:
            raise Exception("Model not trained yet.")

        try:
            anime_idx = list(self.item_user_matrix.index).index(anime_id)
        except ValueError:
            return []

        sim_scores = list(enumerate(self.similarity_matrix[anime_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_k+1]
        similar_animes = [self.item_user_matrix.index[i] for i, _ in sim_scores]
        return similar_animes
