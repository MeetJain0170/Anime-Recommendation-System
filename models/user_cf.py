import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class UserCF:
    def __init__(self, ratings_df):
        """
        Initialize with a DataFrame 'ratings_df' containing:
          - 'username_code': encoded user IDs,
          - 'anime_code': encoded item IDs,
          - 'rating': user-item interaction value.
        """
        self.ratings_df = ratings_df
        self.user_item_matrix = None  # Sparse matrix of user-item interactions
        self.nn_model = None          # NearestNeighbors model
        self.user_index_map = None    # Maps user code to row index in the matrix
        self.index_user_map = None    # Reverse mapping

    def create_user_item_matrix(self):
        # Create a pivot table: rows = users, columns = items, values = ratings
        user_item = self.ratings_df.pivot_table(index='username_code',
                                                columns='anime_code',
                                                values='rating')
        return user_item.fillna(0)

    def train(self):
        # Build the user-item DataFrame and create mapping dictionaries
        user_item_df = self.create_user_item_matrix()
        self.user_index_map = {uid: idx for idx, uid in enumerate(user_item_df.index)}
        self.index_user_map = {idx: uid for uid, idx in self.user_index_map.items()}
        
        # Convert the user-item DataFrame to a sparse matrix to save memory
        self.user_item_matrix = csr_matrix(user_item_df.values)
        
        # Fit a NearestNeighbors model using cosine distance (which is 1 - cosine similarity)
        self.nn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=51, n_jobs=-1)
        self.nn_model.fit(self.user_item_matrix)

    def get_most_similar_neighbors(self, user_id, k_neighbors=50):
        # Return the top k similar neighbors for the given user (excluding the user themself)
        if user_id not in self.user_index_map:
            return []
        user_idx = self.user_index_map[user_id]
        distances, indices = self.nn_model.kneighbors(self.user_item_matrix[user_idx], n_neighbors=k_neighbors + 1)
        neighbor_indices = indices.flatten()[1:]  # Exclude self (first index)
        return [self.index_user_map[idx] for idx in neighbor_indices]

    def recommend(self, user_id, top_k=10):
        if self.user_item_matrix is None or self.nn_model is None:
            raise Exception("Model not trained. Call train() before recommend().")
        
        # Get the nearest neighbors for the user
        neighbors = self.get_most_similar_neighbors(user_id)
        if not neighbors:
            return []
        
        # Reconstruct a (dense) user-item DataFrame for scoring recommendations.
        # This conversion is done only at recommendation time.
        user_item_df = pd.DataFrame.sparse.from_spmatrix(
            self.user_item_matrix,
            index=list(self.user_index_map.keys()),
            columns=self.ratings_df['anime_code'].astype('category').cat.categories
        )
        
        # Get the items this user has already interacted with.
        user_vector = user_item_df.loc[user_id]
        seen_items = user_vector[user_vector > 0].index.tolist()
        
        # Aggregate scores for items (that the user hasn't seen) from the nearest neighbors.
        scores = {}
        for neighbor in neighbors:
            neighbor_vector = user_item_df.loc[neighbor]
            for item in neighbor_vector.index:
                if item in seen_items:
                    continue
                scores[item] = scores.get(item, 0) + neighbor_vector[item]
        
        # Rank candidate items by the aggregated score and return the top recommendations.
        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        recommended_items = [item for item, score in ranked_items[:top_k]]
        return recommended_items
