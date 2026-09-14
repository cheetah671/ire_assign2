import zipfile
from pathlib import Path
txt_path = Path('submissions/EBNERD_LARGE/predictions.txt')
with zipfile.ZipFile('ebnerd_submission_lzma.zip', 'w', compression=zipfile.ZIP_LZMA) as zf:
    zf.write(txt_path, arcname='predictions.txt')
