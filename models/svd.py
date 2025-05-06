# models/svd.py
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

class SVDRecommender:
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df
        self.model = SVD()

    def train(self):
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(self.ratings_df[['username', 'anime_id', 'rating']].dropna(), reader)
        trainset, _ = train_test_split(data, test_size=0.2)
        self.model.fit(trainset)

    def recommend(self, user_id, anime_ids, top_k=10):
        predictions = [self.model.predict(user_id, anime_id) for anime_id in anime_ids]
        predictions.sort(key=lambda x: x.est, reverse=True)
        return [pred.iid for pred in predictions[:top_k]]
