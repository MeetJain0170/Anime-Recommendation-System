# models/neural_cf.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dot, Flatten, Dense, Concatenate
from sklearn.preprocessing import LabelEncoder
import numpy as np

class NeuralCF(Model):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NeuralCF, self).__init__()
        self.user_embedding = Embedding(num_users, embedding_dim, embeddings_initializer='he_normal')
        self.item_embedding = Embedding(num_items, embedding_dim, embeddings_initializer='he_normal')
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(32, activation='relu')
        self.output_layer = Dense(1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        x = Concatenate()([user_vector, item_vector])
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)

class NeuralCFRecommender:
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.model = None

    def train(self, epochs=5):
        # Prepare the data
        df = self.ratings_df[['username', 'anime_id', 'rating']].dropna()
        df['user'] = self.user_encoder.fit_transform(df['username'])
        df['item'] = self.item_encoder.fit_transform(df['anime_id'])

        user_item = df[['user', 'item']].values
        ratings = df['rating'].values

        num_users = df['user'].nunique()
        num_items = df['item'].nunique()

        # Initialize and compile the model
        self.model = NeuralCF(num_users, num_items)
        self.model.compile(optimizer='adam', loss='mse')

        # Train the model
        self.model.fit(user_item, ratings, epochs=epochs, batch_size=64, verbose=1)

    def recommend(self, username, anime_ids, top_k=10):
        # Ensure the model is trained
        if self.model is None:
            print("Model is not trained yet. Please call 'train()' first.")
            return []

        # Encode the user and item
        user_id = self.user_encoder.transform([username])[0]
        item_ids = self.item_encoder.transform(anime_ids)
        
        # Create user-item pairs
        user_item_pairs = np.array([[user_id, item_id] for item_id in item_ids])

        # Predict ratings for each pair
        predictions = self.model.predict(user_item_pairs).flatten()

        # Sort by predicted ratings and return top-k recommendations
        top_indices = predictions.argsort()[::-1][:top_k]
        recommended_items = [anime_ids[i] for i in top_indices]
        return recommended_items
