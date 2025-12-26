Appendix A. Python Code for the Modified Borda Rank Aggregation
```python
import pandas as pd

def modified_borda_mean_rank(method_scores: dict[str, pd.Series]) -> pd.DataFrame:
    df = pd.DataFrame(method_scores)
    ranks = df.rank(ascending=False, method="max")
    ranks["mean_rank"] = ranks.mean(axis=1)
    ranks.insert(0, "feature", ranks.index)
    return ranks.sort_values("mean_rank").reset_index(drop=True)
```