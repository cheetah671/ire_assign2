import pandas as pd
df = pd.DataFrame({
    'impression_id': [6451339, 6451363],
    'article_id': [[9796198,9531110,9796527], [9798906,9791602,9798975]]
})
df.to_parquet('test_pred.parquet')
