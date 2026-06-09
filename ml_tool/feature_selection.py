import pandas as pd
import numpy as np
from typing import Optional
from .feature_analysis import FeatureAnalyzer
from .binning import Binning


class FeatureSelector:
    """按 IV、PSI、缺失率、相关性四个维度筛选特征，记录每个特征的筛选结果"""

    def __init__(
        self,
        iv_threshold: float = 0.001,
        psi_threshold: float = 0.5,
        missing_threshold: float = 0.98,
        corr_threshold: float = 0.97,
        psi_filter: bool = True,
        corr_filter: bool = True,
    ):
        self.iv_threshold = iv_threshold
        self.psi_threshold = psi_threshold
        self.missing_threshold = missing_threshold
        self.corr_threshold = corr_threshold
        self.psi_filter = psi_filter    # False：PSI 只计算展示，不做剔除
        self.corr_filter = corr_filter  # False：相关性只计算展示，不做剔除
        self.report_df: pd.DataFrame = pd.DataFrame()
        self.selected_cols: list = []
        self.dropped_cols: list = []

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        base_df: Optional[pd.DataFrame] = None,
        compare_df: Optional[pd.DataFrame] = None,
        cat_cols: Optional[list] = None,
        n_bins: int = 10,
    ) -> "FeatureSelector":
        cat_cols = cat_cols or []
        feature_cols = [c for c in df.columns if c != target and c not in cat_cols]

        # -------- 缺失率 --------
        analyzer = FeatureAnalyzer(df[feature_cols], cat_cols=[])
        missing_df = analyzer.missing_rate().set_index("特征名")

        # -------- PSI --------
        psi_df = pd.DataFrame(columns=["特征名", "PSI"])
        if base_df is not None and compare_df is not None:
            psi_df = analyzer.psi(base_df, compare_df, cols=feature_cols).set_index("特征名")

        # -------- IV --------
        binning = Binning(n_bins=n_bins)
        valid_for_binning = [
            c for c in feature_cols
            if missing_df.loc[c, "缺失率"] < self.missing_threshold
        ] if not missing_df.empty else feature_cols
        binning.fit(df[valid_for_binning + [target]], target=target, cols=valid_for_binning)
        iv_df = binning.get_iv_summary().set_index("特征名")

        # -------- 相关性 --------
        corr_matrix = df[feature_cols].corr(method="pearson").abs()

        rows = []
        drop_set = set()

        for col in feature_cols:
            miss = missing_df.loc[col, "缺失率"] if col in missing_df.index else np.nan
            psi = psi_df.loc[col, "PSI"] if col in psi_df.index else np.nan
            iv = iv_df.loc[col, "IV"] if col in iv_df.index else np.nan

            reasons = []
            if miss >= self.missing_threshold:
                reasons.append(f"缺失率={miss:.3f}>={self.missing_threshold}")
            if self.psi_filter and not np.isnan(psi) and psi >= self.psi_threshold:
                reasons.append(f"PSI={psi:.3f}>={self.psi_threshold}")
            if np.isnan(iv) or iv < self.iv_threshold:
                reasons.append(f"IV={iv if not np.isnan(iv) else 'N/A'}<{self.iv_threshold}")

            rows.append({
                "特征名": col,
                "缺失率": round(miss, 4) if not np.isnan(miss) else None,
                "PSI": round(psi, 4) if not np.isnan(psi) else None,
                "IV": round(iv, 4) if not np.isnan(iv) else None,
                "_reasons": reasons,
            })

        # -------- 相关性剔除（保留 IV 较高的一个）--------
        # 先把已因其他原因剔除的放入 drop_set
        for r in rows:
            if r["_reasons"]:
                drop_set.add(r["特征名"])

        if self.corr_filter:
            surviving = [r["特征名"] for r in rows if r["特征名"] not in drop_set]
            for i, col_a in enumerate(surviving):
                if col_a in drop_set:
                    continue
                for col_b in surviving[i + 1:]:
                    if col_b in drop_set:
                        continue
                    if col_a not in corr_matrix.index or col_b not in corr_matrix.columns:
                        continue
                    c = corr_matrix.loc[col_a, col_b]
                    if c >= self.corr_threshold:
                        iv_a = iv_df.loc[col_a, "IV"] if col_a in iv_df.index else 0
                        iv_b = iv_df.loc[col_b, "IV"] if col_b in iv_df.index else 0
                        dropped = col_b if iv_a >= iv_b else col_a
                        kept = col_a if dropped == col_b else col_b
                        drop_set.add(dropped)
                        for r in rows:
                            if r["特征名"] == dropped:
                                r["_reasons"].append(f"与{kept}相关系数={c:.3f}>={self.corr_threshold}")

        for r in rows:
            r["是否保留"] = "保留" if not r["_reasons"] else "剔除"
            r["剔除原因"] = "；".join(r["_reasons"]) if r["_reasons"] else ""
            del r["_reasons"]

        self.report_df = pd.DataFrame(rows)[[
            "特征名", "缺失率", "PSI", "IV", "是否保留", "剔除原因"
        ]]
        self.selected_cols = self.report_df[self.report_df["是否保留"] == "保留"]["特征名"].tolist()
        self.dropped_cols = self.report_df[self.report_df["是否保留"] == "剔除"]["特征名"].tolist()
        return self

    def get_selected(self) -> list:
        return self.selected_cols

    def get_report(self) -> pd.DataFrame:
        return self.report_df
