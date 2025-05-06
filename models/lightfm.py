# models/lightfm_model.py
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset

class LightFMRecommender:
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df
        self.model = None
        self.dataset = Dataset()
        self.interactions = None
        self.user_ids = {}
        self.anime_ids = {}

    def train(self):
        df = self.ratings_df[['username', 'anime_id']].dropna()
        self.dataset.fit(df['username'], df['anime_id'])
        self.interactions, _ = self.dataset.build_interactions([(x['username'], x['anime_id']) for _, x in df.iterrows()])
        self.model = LightFM(loss='warp')
        self.model.fit(self.interactions, epochs=10, num_threads=2)

        self.user_ids, self.anime_ids, _, _ = self.dataset.mapping()

    def recommend(self, username, top_k=10):
        if self.model is None:
            raise Exception("Model not trained yet.")

        user_id = self.user_ids.get(username)
        if user_id is None:
            return []

        scores = self.model.predict(user_id, np.arange(len(self.anime_ids)))
        top_items = np.argsort(-scores)[:top_k]
        reverse_anime_ids = {v: k for k, v in self.anime_ids.items()}
        return [reverse_anime_ids[i] for i in top_items]

# code
# print("🔧 Training LightFM Hybrid...")
# lightfm_model = LightFMRecommender(sparse_matrix, anime_df)
# lightfm_model.train()
# lightfm_precision, lightfm_recall, lightfm_ndcg = evaluate_model(lightfm_model, sample_username_code)
# print(f"Precision@K: {lightfm_precision}, Recall@K: {lightfm_recall}, NDCG@K: {lightfm_ndcg}")