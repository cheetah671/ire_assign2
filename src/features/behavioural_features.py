"""
behavioural_features.py — Q1: Click-History & Session Features

Engineers behavioural features from click-logs for both datasets:
1. Click-history features: click count, recency-weighted history
2. Article features: popularity, freshness, category match
3. Enforces behaviour-window boundary: only uses clicks where click_time < impression_time
"""

import numpy as np
import pandas as pd
from datetime import datetime

# Schema columns matching A1 definitions
COL_USER_ID = "user_id"
COL_ARTICLE_ID = "article_id"
COL_IMPRESSION_TIME = "impression_time"
COL_CLICK_TIME = "click_time"
COL_CATEGORY = "category"
COL_PUBLISHED_TIME = "published_time"

class BehaviouralFeatureExtractor:
    def __init__(self, history_df: pd.DataFrame, articles_df: pd.DataFrame):
        self.history = history_df
        self.articles = articles_df
        
        # Precompute global article stats up to the max timestamp if needed,
        # but to strictly avoid leakage, we should compute features per-impression.
        self.article_metadata = self.articles.set_index(COL_ARTICLE_ID)

    def extract_features(self, impressions_df: pd.DataFrame, half_life_days: float = 7.0) -> pd.DataFrame:
        """
        Extract features for a given set of impressions.
        impressions_df must contain: impression_id, user_id, impression_time, article_id (candidate)
        Returns a DataFrame with the same number of rows and added feature columns.
        """
        # Ensure time is datetime
        impressions_df = impressions_df.copy()
        impressions_df[COL_IMPRESSION_TIME] = pd.to_datetime(impressions_df[COL_IMPRESSION_TIME])
        
        # Features to collect
        features = []
        
        # Group by impression to process candidates together
        grouped = impressions_df.groupby("impression_id", sort=False)
        
        for imp_id, group in grouped:
            uid = group[COL_USER_ID].iloc[0]
            imp_time = group[COL_IMPRESSION_TIME].iloc[0]
            candidate_ids = group[COL_ARTICLE_ID].values
            
            # 1. Enforce Behavioural-Window Boundary (Q1.4)
            # Only use history strictly before the impression time
            user_hist = self.history[(self.history[COL_USER_ID] == uid) & 
                                     (self.history[COL_CLICK_TIME] < imp_time)]
            
            # Click count
            click_count = len(user_hist)
            
            # Recency-weighted history (exponential decay)
            # Weight = exp(-ln(2) * (imp_time - click_time).days / half_life)
            hist_weights = []
            user_hist_categories = set()
            if click_count > 0:
                time_diffs = (imp_time - user_hist[COL_CLICK_TIME]).dt.total_seconds() / (3600 * 24)
                hist_weights = np.exp(-np.log(2) * time_diffs / half_life_days)
                
                # Fetch categories for user history
                clicked_articles = user_hist[COL_ARTICLE_ID].values
                valid_clicked = [a for a in clicked_articles if a in self.article_metadata.index]
                if valid_clicked:
                    user_hist_categories = set(self.article_metadata.loc[valid_clicked, COL_CATEGORY].dropna().values)

            imp_features = []
            for cand_id in candidate_ids:
                f_dict = {
                    "impression_id": imp_id,
                    "article_id": cand_id,
                    "user_click_count": click_count,
                    "user_hist_recency_sum": np.sum(hist_weights) if click_count > 0 else 0.0,
                }
                
                # Article features
                if cand_id in self.article_metadata.index:
                    cand_meta = self.article_metadata.loc[cand_id]
                    cand_cat = cand_meta.get(COL_CATEGORY, "")
                    pub_time = cand_meta.get(COL_PUBLISHED_TIME, pd.NaT)
                    
                    f_dict["category_match"] = 1 if cand_cat in user_hist_categories else 0
                    
                    if pd.notna(pub_time):
                        pub_time = pd.to_datetime(pub_time)
                        if pub_time.tzinfo is not None:
                            pub_time = pub_time.tz_localize(None)
                        freshness_days = (imp_time - pub_time).total_seconds() / (3600 * 24)
                        f_dict["freshness_days"] = max(0, freshness_days)
                    else:
                        f_dict["freshness_days"] = -1 # missing
                else:
                    f_dict["category_match"] = 0
                    f_dict["freshness_days"] = -1
                
                imp_features.append(f_dict)
                
            features.extend(imp_features)
            
        features_df = pd.DataFrame(features)
        
        # Merge back to original df to preserve order and all columns
        merged = pd.merge(impressions_df, features_df, on=["impression_id", "article_id"], how="left")
        return merged
