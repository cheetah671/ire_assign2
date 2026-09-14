import zipfile
from pathlib import Path
import time

txt_path = Path('submissions/EBNERD_LARGE/predictions.txt')

print("Compressing with BZIP2...")
t0 = time.time()
with zipfile.ZipFile('ebnerd_bzip2.zip', 'w', compression=zipfile.ZIP_BZIP2) as zf:
    zf.write(txt_path, arcname='predictions.txt')
print(f"BZIP2 done in {time.time()-t0:.1f}s")

print("Compressing with LZMA...")
t0 = time.time()
with zipfile.ZipFile('ebnerd_lzma.zip', 'w', compression=zipfile.ZIP_LZMA) as zf:
    zf.write(txt_path, arcname='predictions.txt')
print(f"LZMA done in {time.time()-t0:.1f}s")
