import pandas as pd
import numpy as np
from typing import Optional


class Binning:
    """等频10分箱，WOE/IV 计算，特征转换"""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self._bin_edges: dict = {}   # col -> bin edges
        self._woe_map: dict = {}     # col -> {bin_label -> woe}
        self.iv_summary: pd.DataFrame = pd.DataFrame()

    def fit(self, df: pd.DataFrame, target: str, cols: Optional[list] = None) -> "Binning":
        """拟合分箱边界和 WOE/IV，target 为0/1二分类标签列"""
        cols = cols or [c for c in df.columns if c != target]
        total_bad = (df[target] == 1).sum()
        total_good = (df[target] == 0).sum()

        iv_rows = []
        for col in cols:
            sub = df[[col, target]].dropna(subset=[col])
            if sub.empty or sub[col].nunique() < 2:
                continue
            try:
                _, edges = pd.qcut(sub[col], q=self.n_bins, retbins=True, duplicates="drop")
            except Exception:
                continue
            edges[0] = -np.inf
            edges[-1] = np.inf
            self._bin_edges[col] = edges

            sub = sub.copy()
            sub["_bin"] = pd.cut(sub[col], bins=edges, include_lowest=True)
            grp = sub.groupby("_bin", observed=True)[target].agg(["sum", "count"])
            grp.columns = ["坏样本数", "总样本数"]
            grp["好样本数"] = grp["总样本数"] - grp["坏样本数"]
            grp["坏样本占比"] = (grp["坏样本数"] / total_bad).clip(lower=1e-6)
            grp["好样本占比"] = (grp["好样本数"] / total_good).clip(lower=1e-6)
            grp["WOE"] = np.log(grp["坏样本占比"] / grp["好样本占比"]).round(4)
            grp["IV_bin"] = ((grp["坏样本占比"] - grp["好样本占比"]) * grp["WOE"]).round(6)
            self._woe_map[col] = grp["WOE"].to_dict()

            iv_total = round(grp["IV_bin"].sum(), 4)
            grp.index = grp.index.astype(str)
            iv_rows.append({"特征名": col, "IV": iv_total, "分箱详情": grp.reset_index().rename(columns={"_bin": "分箱区间"})})

        self.iv_summary = pd.DataFrame([{"特征名": r["特征名"], "IV": r["IV"]} for r in iv_rows])
        self._iv_details = {r["特征名"]: r["分箱详情"] for r in iv_rows}
        return self

    def transform(self, df: pd.DataFrame, cols: Optional[list] = None, mode: str = "bin_index") -> pd.DataFrame:
        """将特征转换为分箱编号(mode='bin_index')或 WOE 值(mode='woe')"""
        out = df.copy()
        cols = cols or list(self._bin_edges.keys())
        for col in cols:
            if col not in self._bin_edges:
                continue
            edges = self._bin_edges[col]
            binned = pd.cut(out[col], bins=edges, include_lowest=True, labels=False)
            if mode == "woe":
                bin_labels = pd.cut(out[col], bins=edges, include_lowest=True)
                woe_map = self._woe_map.get(col, {})
                out[col] = bin_labels.map(woe_map)
            else:
                out[col] = binned
        return out

    def fit_transform(self, df: pd.DataFrame, target: str, cols: Optional[list] = None, mode: str = "bin_index") -> pd.DataFrame:
        self.fit(df, target, cols)
        return self.transform(df, cols, mode)

    def get_iv_details(self, col: str) -> pd.DataFrame:
        return self._iv_details.get(col, pd.DataFrame())

    def get_iv_summary(self) -> pd.DataFrame:
        return self.iv_summary.sort_values("IV", ascending=False).reset_index(drop=True)
