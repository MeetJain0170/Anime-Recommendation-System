import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler  # Missing import
from sklearn.feature_extraction.text import TfidfVectorizer  # Missing import
from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
from surprise.model_selection import GridSearchCV
import tensorflow as tf
from lightfm import LightFM
import implicit
from scipy.sparse import csr_matrix, hstack  # Added hstack

# Load the preprocessed datasets
ratings_df = pd.read_csv('ratings.csv')
anime_df = pd.read_csv('anime.csv')
users_df = pd.read_csv('users.csv')

# Split data into train and test sets
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

class AnimeRecommenderSystem:
    def __init__(self):
        self.models = {}
        self.metrics = {}
    
    def create_user_item_matrix(self, df):
        return df.pivot(index='username', columns='anime_id', values='rating').fillna(0)
    
    def calculate_metrics(self, y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae}

    def user_based_cf(self, train_matrix):
        """Enhanced User-Based Collaborative Filtering"""
        print("Training User-Based CF...")
        
        # Create enhanced user features
        user_features = users_df[['stats_mean_score', 'stats_rewatched', 'stats_episodes', 
                                'activity_ratio', 'engagement_score']].values
        
        # Normalize user features
        scaler = StandardScaler()
        normalized_user_features = scaler.fit_transform(user_features)
        
        # Combine rating matrix with user features
        enhanced_user_matrix = np.hstack([train_matrix.values, normalized_user_features])
        
        # Calculate similarity
        user_similarity = cosine_similarity(enhanced_user_matrix)
        self.models['user_cf'] = user_similarity
        return user_similarity

    def item_based_cf(self, train_matrix):
        """Enhanced Item-Based Collaborative Filtering"""
        print("Training Item-Based CF...")
        
        # Create enhanced item features
        item_features = anime_df[['score', 'scored_by', 'rank', 'popularity', 
                                'members', 'favorites']].values
        
        # Normalize item features
        scaler = StandardScaler()
        normalized_item_features = scaler.fit_transform(item_features)
        
        # Combine rating matrix with item features
        enhanced_item_matrix = np.hstack([train_matrix.T.values, normalized_item_features])
        
        # Calculate similarity
        item_similarity = cosine_similarity(enhanced_item_matrix)
        self.models['item_cf'] = item_similarity
        return item_similarity

    def matrix_factorization_svd(self, train_data, test_data):
        """Enhanced SVD Matrix Factorization"""
        print("Training SVD Model...")
        
        # Add user and anime features to the training data
        enhanced_train_data = train_data.merge(
            users_df[['username', 'stats_mean_score', 'engagement_score']], 
            on='username'
        ).merge(
            anime_df[['anime_id', 'score', 'popularity']], 
            on='anime_id'
        )
        
        reader = Reader(rating_scale=(1, 10))
        train_set = Dataset.load_from_df(
            enhanced_train_data[['username', 'anime_id', 'rating', 
                            'stats_mean_score', 'engagement_score', 
                            'score', 'popularity']], 
            reader
        )
        
        # Enhanced parameter tuning
        param_grid = {
            'n_factors': [50, 100, 150],
            'n_epochs': [20, 30, 40],
            'lr_all': [0.003, 0.005, 0.01],
            'reg_all': [0.02, 0.04, 0.06]
        }
        
        gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
        gs.fit(train_set)
        
        best_params = gs.best_params['rmse']
        svd = SVD(**best_params)
        svd.fit(train_set.build_full_trainset())
        
        self.models['svd'] = svd
        return svd

    def neural_cf(self, train_data):
        """Enhanced Neural Collaborative Filtering"""
        print("Training Neural CF Model...")
        
        # Create embeddings
        n_users = len(train_data['username'].unique())
        n_items = len(train_data['anime_id'].unique())
        n_factors = 100  # Increased from 50
        
        # Enhanced inputs
        user_input = tf.keras.layers.Input(shape=(1,))
        item_input = tf.keras.layers.Input(shape=(1,))
        user_metadata_input = tf.keras.layers.Input(shape=(5,))  # User features
        item_metadata_input = tf.keras.layers.Input(shape=(6,))  # Anime features
        
        # Embeddings
        user_embedding = tf.keras.layers.Embedding(n_users, n_factors)(user_input)
        item_embedding = tf.keras.layers.Embedding(n_items, n_factors)(item_input)
        
        user_vec = tf.keras.layers.Flatten()(user_embedding)
        item_vec = tf.keras.layers.Flatten()(item_embedding)
        
        # Process metadata
        user_metadata = tf.keras.layers.Dense(32, activation='relu')(user_metadata_input)
        item_metadata = tf.keras.layers.Dense(32, activation='relu')(item_metadata_input)
        
        # Combine all features
        concat = tf.keras.layers.Concatenate()([user_vec, item_vec, user_metadata, item_metadata])
        
        # Deeper network
        dense1 = tf.keras.layers.Dense(256, activation='relu')(concat)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(128, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
        dense3 = tf.keras.layers.Dense(64, activation='relu')(dropout2)
        
        output = tf.keras.layers.Dense(1)(dense3)
        
        model = tf.keras.Model(
            inputs=[user_input, item_input, user_metadata_input, item_metadata_input],
            outputs=output
        )
        
        model.compile(
            loss='mse',
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=['mae', 'mse']
        )
        
        self.models['neural_cf'] = model
        return model

    def content_based_filtering(self):
        """Content-Based Filtering"""
        print("Training Content-Based Model...")
        
        # Create feature matrix from anime attributes
        feature_cols = [
            # Basic features
            'genre', 'type', 'source', 'studio', 'producer',
            # Numerical features
            'score', 'duration_min', 'members', 'favorites'
        ]
        
        # TF-IDF Vectorization for text features
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Combine text features
        anime_df['text_features'] = anime_df[['genre', 'type', 'source', 'studio', 'producer']].apply(
            lambda x: ' '.join(x.astype(str)), axis=1
        )
        
        tfidf = TfidfVectorizer(stop_words='english')
        text_features = tfidf.fit_transform(anime_df['text_features'])
        
        # Combine with numerical features
        numerical_features = anime_df[['score', 'duration_min', 'members', 'favorites']].values
        one_hot_features = anime_df[[col for col in anime_df.columns if col.startswith(('genre_', 'type_', 'source_'))]].values
        
        # Normalize numerical features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numerical_features = scaler.fit_transform(numerical_features)
        
        # Combine all features
        from scipy.sparse import hstack
        anime_features = hstack([text_features, numerical_features, one_hot_features])
        
        # Calculate similarity matrix
        content_similarity = cosine_similarity(anime_features)
        
        self.models['content_based'] = content_similarity
        return content_similarity

    def hybrid_lightfm(self, train_data):
        """Enhanced Hybrid Model using LightFM"""
        print("Training LightFM Hybrid Model...")
        
        # Create LightFM dataset with more features
        lightfm_data = LightFM(
            learning_rate=0.05,
            loss='warp',
            no_components=150,  # Increased from default
            item_alpha=1e-6
        )
        
        # Enhanced user features
        user_features = users_df[[
            'gender_Female', 'gender_Male', 'gender_Non-Binary',
            'stats_mean_score', 'stats_rewatched', 'stats_episodes',
            'activity_ratio', 'engagement_score'
        ]].values
        
        # Enhanced anime features
        anime_features = anime_df[[
            # Genre features
            *[col for col in anime_df.columns if col.startswith('genre_')],
            # Type features
            *[col for col in anime_df.columns if col.startswith('type_')],
            # Source features
            *[col for col in anime_df.columns if col.startswith('source_')],
            # Normalized numerical features
            'normalized_score', 'popularity_score'
        ]].values
        
        # Train model with more epochs and better monitoring
        lightfm_data.fit(
            interactions=train_data,
            user_features=user_features,
            item_features=anime_features,
            epochs=50,  # Increased from 30
            num_threads=4,
            verbose=True
        )
        
        self.models['lightfm'] = lightfm_data
        return lightfm_data

    def implicit_als(self, train_data):
        """Enhanced Implicit ALS Model"""
        print("Training Implicit ALS Model...")
        
        # Convert ratings to weighted implicit feedback
        confidence = train_data['rating'].copy()
        confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min()) * 40 + 1
        
        sparse_ratings = csr_matrix((
            confidence.astype(float),
            (
                train_data['username'].astype(np.int32),
                train_data['anime_id'].astype(np.int32)
            )
        ))
        
        # Enhanced model parameters
        model = implicit.als.AlternatingLeastSquares(
            factors=150,  # Increased from 100
            regularization=0.1,
            iterations=100,  # Increased from 50
            calculate_training_loss=True,
            num_threads=4
        )
        
        # Fit model with confidence weights
        model.fit(sparse_ratings)
        
        self.models['implicit_als'] = model
        return model

def evaluate_models():
    recommender = AnimeRecommenderSystem()
    
    # Create user-item matrix
    train_matrix = recommender.create_user_item_matrix(train_data)
    
    # Train and evaluate each model
    models_to_train = [
        recommender.user_based_cf,
        recommender.item_based_cf,
        recommender.matrix_factorization_svd,
        recommender.neural_cf,
        recommender.content_based_filtering,
        recommender.hybrid_lightfm,
        recommender.implicit_als
    ]
    
    results = {}
    for model_func in models_to_train:
        model_name = model_func.__name__
        print(f"\nTraining {model_name}...")
        
        try:
            model = model_func(train_matrix if 'cf' in model_name else train_data)
            
            # Make predictions on test set
            test_predictions = model.predict(test_data)
            
            # Calculate metrics
            metrics = recommender.calculate_metrics(
                test_data['rating'],
                test_predictions
            )
            
            results[model_name] = metrics
            print(f"{model_name} metrics:", metrics)
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
    
    return results

# Run evaluation
results = evaluate_models()

# Print final results
print("\nFinal Model Comparison:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")