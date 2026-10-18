# Rank-50 UTXSS from the existing rank-75 factorization

The current UTXSS artifact was generated from LibriSpeech `train-clean-100`,
with speaker `103` as the anchor/source:

```text
exp/content_factorization/librispeech_train-clean-100/rank_75/src_103/UTXSS.npy
```

It has shape `(1024, 75)`.  The rank-50 version for the same factorization is
its leading 50 columns:

```python
from pathlib import Path
import numpy as np

src = Path(
    "exp/content_factorization/librispeech_train-clean-100/rank_75/src_103/UTXSS.npy"
)
dst = src.parents[1] / "rank_50" / "src_103" / "UTXSS.npy"
dst.parent.mkdir(parents=True, exist_ok=True)
np.save(dst, np.load(src)[:, :50])
```

This is valid despite UTXSS being computed with least squares.  The fit is

\[
\min_A \lVert XS A - U_{\mathrm{rep}} \rVert_F^2,
\]

which separates over the output columns.  Thus columns 51--75 do not affect
the least-squares solutions for columns 1--50.  The subsequent multiplication
by `diag(S)` is also column-wise.  Consequently, truncating the rank-75
UTXSS to its first 50 columns is equivalent to recomputing rank 50 from the
same factorization inputs, apart from negligible floating-point differences.

This statement applies only to the same data, source speaker, and SVD
ordering; it is not a substitute for a separately trained factorization using
different data or settings.
