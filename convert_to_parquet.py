import pandas as pd
from tqdm import tqdm

def txt_to_parquet(txt_path, pq_path):
    impressions = []
    ranks_list = []
    
    with open(txt_path, 'r') as f:
        for line in tqdm(f, desc="Reading"):
            line = line.strip()
            if not line: continue
            imp_str, ranks_str = line.split(" [")
            ranks_str = ranks_str.rstrip("]")
            
            imp_id = int(imp_str)
            ranks = [int(x) for x in ranks_str.split(",")]
            
            impressions.append(imp_id)
            ranks_list.append(ranks)
            
    df = pd.DataFrame({
        "impression_id": impressions,
        "prediction": ranks_list
    })
    df.to_parquet(pq_path, engine="pyarrow")
    print(f"Saved to {pq_path}")

txt_to_parquet("submissions/EBNERD_LARGE/predictions.txt", "submissions/EBNERD_LARGE/predictions.parquet")
