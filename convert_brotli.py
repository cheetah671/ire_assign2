import pandas as pd
df = pd.read_parquet("submissions/EBNERD_LARGE/predictions.parquet")
df.to_parquet("submissions/EBNERD_LARGE/predictions_brotli.parquet", compression="brotli")
