"""
reranker.py — Q2: Re-Ranker

Trains a GBDT (LightGBM) over engineered behavioural features.
"""

import logging
import numpy as np
import pandas as pd
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

logger = logging.getLogger("reranker")

class LightGBMReranker:
    def __init__(self, feature_cols: list):
        self.feature_cols = feature_cols
        self.model = None

    def train(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None):
        """
        Train the LightGBM model.
        df_train and df_val must contain columns in self.feature_cols and a 'label' column.
        For ranking, we also need group information (number of candidates per impression).
        """
        if lgb is None:
            raise ImportError("lightgbm is not installed. Please install it using `pip install lightgbm`.")

        logger.info(f"Training LightGBM on {len(df_train)} samples with {len(self.feature_cols)} features...")
        
        # Sort by impression_id to properly construct group information
        df_train = df_train.sort_values("impression_id")
        X_train = df_train[self.feature_cols].fillna(0)
        y_train = df_train["label"]
        group_train = df_train.groupby("impression_id").size().values

        train_data = lgb.Dataset(X_train, label=y_train, group=group_train)

        valid_sets = [train_data]
        valid_names = ['train']

        if df_val is not None:
            df_val = df_val.sort_values("impression_id")
            X_val = df_val[self.feature_cols].fillna(0)
            y_val = df_val["label"]
            group_val = df_val.groupby("impression_id").size().values
            val_data = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        # Baseline parameters
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'random_state': 42
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(stopping_rounds=10)] if df_val is not None else []
        )
        logger.info("Training completed.")
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate scores for the given dataframe.
        """
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        
        X = df[self.feature_cols].fillna(0)
        scores = self.model.predict(X)
        return scores
