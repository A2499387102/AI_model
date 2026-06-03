import pandas as pd
import numpy as np
from typing import Optional


class FeatureAnalyzer:
    """特征分析：描述性统计、缺失率、一值率、分位数、PSI、相关性"""

    def __init__(self, df: pd.DataFrame, cat_cols: Optional[list] = None):
        self.df = df.copy()
        self.cat_cols = cat_cols or []
        self.num_cols = [c for c in df.columns if c not in self.cat_cols]

    def missing_rate(self) -> pd.DataFrame:
        total = len(self.df)
        missing = self.df.isnull().sum()
        return pd.DataFrame({
            "特征名": missing.index,
            "缺失数": missing.values,
            "缺失率": (missing / total).round(4).values,
        })

    def single_value_rate(self) -> pd.DataFrame:
        total = len(self.df)
        rows = []
        for col in self.df.columns:
            top_val = self.df[col].value_counts(dropna=False).iloc[0] if total > 0 else 0
            top_name = self.df[col].value_counts(dropna=False).index[0] if total > 0 else None
            rows.append({
                "特征名": col,
                "最高频值": top_name,
                "一值率": round(top_val / total, 4),
            })
        return pd.DataFrame(rows)

    def quantile_stats(self) -> pd.DataFrame:
        """数值型特征描述性统计和分位数，类别型跳过"""
        num_df = self.df[self.num_cols]
        if num_df.empty:
            return pd.DataFrame()
        desc = num_df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
        desc.index.name = "特征名"
        desc.columns = [
            "样本数", "均值", "标准差", "最小值",
            "1%分位", "5%分位", "25%分位", "中位数",
            "75%分位", "95%分位", "99%分位", "最大值",
        ]
        return desc.reset_index()

    def psi(
        self,
        base_df: pd.DataFrame,
        compare_df: pd.DataFrame,
        bins: int = 10,
        cols: Optional[list] = None,
    ) -> pd.DataFrame:
        """计算基准组与对照组的 PSI（仅数值型特征）"""
        cols = cols or self.num_cols
        results = []
        for col in cols:
            if col not in base_df.columns or col not in compare_df.columns:
                continue
            b = base_df[col].dropna()
            c = compare_df[col].dropna()
            if b.empty or c.empty:
                results.append({"特征名": col, "PSI": np.nan})
                continue
            _, bin_edges = pd.qcut(b, q=bins, retbins=True, duplicates="drop")
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            b_cnt = pd.cut(b, bins=bin_edges).value_counts(sort=False)
            c_cnt = pd.cut(c, bins=bin_edges).value_counts(sort=False)
            b_pct = (b_cnt / len(b)).clip(lower=1e-6)
            c_pct = (c_cnt / len(c)).clip(lower=1e-6)
            psi_val = ((c_pct - b_pct) * np.log(c_pct / b_pct)).sum()
            results.append({"特征名": col, "PSI": round(psi_val, 4)})
        return pd.DataFrame(results)

    def correlation(self, method: str = "pearson", threshold: float = 0.0) -> pd.DataFrame:
        """特征间相关性矩阵（仅数值型），返回高相关特征对"""
        num_df = self.df[self.num_cols]
        if num_df.empty:
            return pd.DataFrame()
        corr_matrix = num_df.corr(method=method)
        if threshold <= 0.0:
            return corr_matrix
        rows = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr_matrix.iloc[i, j]
                if abs(val) >= threshold:
                    rows.append({
                        "特征A": cols[i],
                        "特征B": cols[j],
                        "相关系数": round(val, 4),
                    })
        return pd.DataFrame(rows).sort_values("相关系数", ascending=False, key=abs) if rows else pd.DataFrame(columns=["特征A", "特征B", "相关系数"])

    def full_analysis(
        self,
        base_df: Optional[pd.DataFrame] = None,
        compare_df: Optional[pd.DataFrame] = None,
        psi_bins: int = 10,
    ) -> dict:
        """一次性返回全部分析结果"""
        result = {
            "缺失率": self.missing_rate(),
            "一值率": self.single_value_rate(),
            "分位数统计": self.quantile_stats(),
            "相关性矩阵": self.correlation(),
        }
        if base_df is not None and compare_df is not None:
            result["PSI"] = self.psi(base_df, compare_df, bins=psi_bins)
        return result
