import os
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr


def _ks_auc(y_true, y_score, weights=None):
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=weights)
        auc = roc_auc_score(y_true, y_score, sample_weight=weights)
        if auc < 0.5:
            auc = 1 - auc
        ks = round(float(np.abs(tpr - fpr).max()), 4)
        return ks, round(float(auc), 4)
    except Exception:
        return None, None


def _mape(y_true, y_pred):
    """平均绝对百分比误差，跳过真实值为 0 的样本"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return None
    return round(float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100), 4)


def _decile_table(y_true, y_score, weights=None, n_bins: int = 10) -> pd.DataFrame:
    """按预测分数等频分箱，统计每箱坏样本率、样本占比、Lift"""
    total_n = len(y_true)
    total_bad = int(np.array(y_true).sum())
    overall_bad_rate = total_bad / total_n if total_n > 0 else 1e-6

    df = pd.DataFrame({"真实值": np.array(y_true), "预测分数": np.array(y_score)})
    df["权重"] = np.array(weights) if weights is not None else 1.0
    try:
        df["分箱"] = pd.qcut(df["预测分数"], q=n_bins, duplicates="drop", labels=False) + 1
    except ValueError:
        df["分箱"] = 1

    rows = []
    for bin_id, g in df.groupby("分箱"):
        n = len(g)
        bad = int((g["真实值"] * g["权重"]).sum())
        bad_rate = bad / n if n > 0 else 0
        lift = round(bad_rate / overall_bad_rate, 4) if overall_bad_rate > 0 else None
        rows.append({
            "分箱": int(bin_id),
            "样本数": n,
            "样本占比": round(n / total_n, 4),
            "坏样本数": bad,
            "加权样本数": round(float(g["权重"].sum()), 2),
            "最低分": round(float(g["预测分数"].min()), 4),
            "最高分": round(float(g["预测分数"].max()), 4),
            "坏样本率": round(bad_rate, 4),
            "Lift": lift,
        })
    if not rows:
        return pd.DataFrame(columns=["分箱", "样本数", "样本占比", "坏样本数", "加权样本数",
                                     "最低分", "最高分", "坏样本率", "Lift"])
    return pd.DataFrame(rows)


def _bucket_table(y_true, y_pred, n_bins: int = 10) -> pd.DataFrame:
    """按预测值等频分桶，统计每桶均值误差、MAPE、样本占比"""
    total_n = len(y_true)
    df = pd.DataFrame({"真实值": np.array(y_true), "预测值": np.array(y_pred)})
    try:
        df["分桶"] = pd.qcut(df["预测值"], q=n_bins, duplicates="drop", labels=False) + 1
    except ValueError:
        df["分桶"] = 1

    rows = []
    for bin_id, g in df.groupby("分桶"):
        n = len(g)
        mae_val  = round(float(mean_absolute_error(g["真实值"].values, g["预测值"].values)), 4)
        mape_val = _mape(g["真实值"].values, g["预测值"].values)
        rows.append({
            "分桶": int(bin_id),
            "样本数": n,
            "样本占比": round(n / total_n, 4),
            "真实均值": round(float(g["真实值"].mean()), 4),
            "预测均值": round(float(g["预测值"].mean()), 4),
            "预测最低值": round(float(g["预测值"].min()), 4),
            "预测最高值": round(float(g["预测值"].max()), 4),
            "MAE": mae_val,
            "MAPE(%)": mape_val,
        })
    if not rows:
        return pd.DataFrame(columns=["分桶", "样本数", "样本占比", "真实均值", "预测均值",
                                     "预测最低值", "预测最高值", "MAE", "MAPE(%)", "误差(预测-真实)"])
    grp = pd.DataFrame(rows)
    grp["误差(预测-真实)"] = (grp["预测均值"] - grp["真实均值"]).round(4)
    return grp


class ReportGenerator:
    """生成二分类 / 回归模型评估报告，以及特征分箱报告，输出 Excel 和 HTML"""

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  二分类报告
    # ------------------------------------------------------------------ #
    def classification_report(
        self,
        datasets: dict,
        filename: str = "分类模型评估报告",
        month_col_data: Optional[dict] = None,
    ) -> dict:
        """
        datasets: {"数据集名": (y_true, y_score[, weight]), ...}
        month_col_data: {"数据集名": df_with_month_col} —— df 须含 'month'/'ym' 等月份列
                        key 须与 datasets 中的名称一致，且 df 行与 y_true 对齐
                        若为 None 则不生成按月 sheet
        """
        summary_rows = []
        sheets = {}   # sheet_name -> DataFrame

        for name, vals in datasets.items():
            y_true, y_score = np.array(vals[0]), np.array(vals[1])
            w = vals[2] if len(vals) > 2 else None
            ks, auc = _ks_auc(y_true, y_score, w)
            summary_rows.append({
                "数据集": name, "KS": ks, "AUC": auc,
                "样本量": len(y_true),
                "坏样本数": int(y_true.sum()),
                "坏样本率": round(float(y_true.mean()), 4),
            })
            sheets[f"{name}_分箱分析"] = _decile_table(y_true, y_score, w)

        # 按月 KS sheet
        if month_col_data:
            ks_month_rows = []
            for name, mdf in month_col_data.items():
                if name not in datasets:
                    continue
                vals = datasets[name]
                y_true, y_score = np.array(vals[0]), np.array(vals[1])
                mdf = mdf.reset_index(drop=True)
                month_col = _detect_month_col(mdf)
                if month_col is None:
                    continue
                tmp = pd.DataFrame({"y": y_true, "s": y_score, "month": mdf[month_col].values})
                for month_val, g in tmp.groupby("month", sort=True):
                    if g["y"].nunique() < 2:
                        continue
                    ks_m, auc_m = _ks_auc(g["y"].values, g["s"].values)
                    ks_month_rows.append({
                        "数据集": name, month_col: month_val,
                        "样本数": len(g), "坏样本数": int(g["y"].sum()),
                        "坏样本率": round(float(g["y"].mean()), 4),
                        "KS": ks_m, "AUC": auc_m,
                    })
            if ks_month_rows:
                sheets["按月KS_AUC"] = pd.DataFrame(ks_month_rows)

            # 按月分箱 sheet（每个数据集一个 sheet）
            for name, mdf in month_col_data.items():
                if name not in datasets:
                    continue
                vals = datasets[name]
                y_true, y_score = np.array(vals[0]), np.array(vals[1])
                mdf = mdf.reset_index(drop=True)
                month_col = _detect_month_col(mdf)
                if month_col is None:
                    continue
                tmp = pd.DataFrame({"y": y_true, "s": y_score, "month": mdf[month_col].values})
                month_bin_rows = []
                for month_val, g in tmp.groupby("month", sort=True):
                    bin_df = _decile_table(g["y"].values, g["s"].values)
                    bin_df.insert(0, month_col, month_val)
                    month_bin_rows.append(bin_df)
                if month_bin_rows:
                    key = f"{name}_按月分箱"[:31]
                    sheets[key] = pd.concat(month_bin_rows, ignore_index=True)

        summary_df = pd.DataFrame(summary_rows)
        excel_path = os.path.join(self.output_dir, f"{filename}.xlsx")
        html_path  = os.path.join(self.output_dir, f"{filename}.html")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="汇总指标", index=False)
            for sname, sdf in sheets.items():
                sdf.to_excel(writer, sheet_name=sname[:31], index=False)

        html = self._build_clf_html(summary_df, sheets, filename)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return {"summary": summary_df, "sheets": sheets,
                "excel_path": excel_path, "html_path": html_path}

    # ------------------------------------------------------------------ #
    #  回归报告
    # ------------------------------------------------------------------ #
    def regression_report(
        self,
        datasets: dict,
        filename: str = "回归模型评估报告",
        n_bins: int = 10,
        month_col_data: Optional[dict] = None,
    ) -> dict:
        """
        datasets: {"数据集名": (y_true, y_pred), ...}
        month_col_data: {"数据集名": df_with_month_col}
        """
        summary_rows = []
        sheets = {}

        for name, (y_true, y_pred) in datasets.items():
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            rmse = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
            mae  = round(float(mean_absolute_error(y_true, y_pred)), 4)
            r2   = round(float(r2_score(y_true, y_pred)), 4)
            mape = _mape(y_true, y_pred)
            pearson_r,  pearson_p  = pearsonr(y_true, y_pred)
            spearman_r, spearman_p = spearmanr(y_true, y_pred)
            summary_rows.append({
                "数据集": name, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE(%)": mape,
                "Pearson相关系数":  round(float(pearson_r),  4),
                "Pearson_p值":      round(float(pearson_p),  4),
                "Spearman相关系数": round(float(spearman_r), 4),
                "Spearman_p值":     round(float(spearman_p), 4),
                "样本量": len(y_true),
            })
            sheets[f"{name}_分桶分析"] = _bucket_table(y_true, y_pred, n_bins)

        # 按月分桶 sheet
        if month_col_data:
            for name, mdf in month_col_data.items():
                if name not in datasets:
                    continue
                y_true_all, y_pred_all = np.array(datasets[name][0]), np.array(datasets[name][1])
                mdf = mdf.reset_index(drop=True)
                month_col = _detect_month_col(mdf)
                if month_col is None:
                    continue
                tmp = pd.DataFrame({"y": y_true_all, "p": y_pred_all,
                                    "month": mdf[month_col].values})
                month_bucket_rows = []
                for month_val, g in tmp.groupby("month", sort=True):
                    bdf = _bucket_table(g["y"].values, g["p"].values, n_bins)
                    bdf.insert(0, month_col, month_val)
                    month_bucket_rows.append(bdf)
                if month_bucket_rows:
                    key = f"{name}_按月分桶"[:31]
                    sheets[key] = pd.concat(month_bucket_rows, ignore_index=True)

        summary_df = pd.DataFrame(summary_rows)
        excel_path = os.path.join(self.output_dir, f"{filename}.xlsx")
        html_path  = os.path.join(self.output_dir, f"{filename}.html")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="汇总指标", index=False)
            for sname, sdf in sheets.items():
                sdf.to_excel(writer, sheet_name=sname[:31], index=False)

        html = self._build_reg_html(summary_df, sheets, filename)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return {"summary": summary_df, "sheets": sheets,
                "excel_path": excel_path, "html_path": html_path}

    # ------------------------------------------------------------------ #
    #  特征筛选 Excel 报告
    # ------------------------------------------------------------------ #
    def feature_selection_report(
        self,
        feature_report_df: pd.DataFrame,
        analysis_results: Optional[dict] = None,
        filename: str = "特征筛选报告",
    ) -> str:
        excel_path = os.path.join(self.output_dir, f"{filename}.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            feature_report_df.to_excel(writer, sheet_name="特征筛选结果", index=False)
            if analysis_results:
                for sheet_name, df in analysis_results.items():
                    if isinstance(df, pd.DataFrame) and not df.empty:
                        df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        return excel_path

    # ------------------------------------------------------------------ #
    #  特征分箱报告（按 dataset 分别输出）
    # ------------------------------------------------------------------ #
    def feature_bin_report(
        self,
        df: pd.DataFrame,
        target: str,
        features: list,
        dataset_col: str = "dataset",
        binning=None,
        n_bins: int = 10,
        filename: str = "特征分箱报告",
    ) -> str:
        """
        按 dataset 列（train/test/oot）分别输出每个特征的分箱统计：
        各箱坏样本占比、样本占比

        参数
        ----
        df          : 含特征列、target 列、dataset_col 列的完整 DataFrame
        target      : 二分类标签列名
        features    : 需要输出分箱图的特征列表
        dataset_col : 区分 train/test/oot 的列名，默认 'dataset'
        binning     : 已拟合的 Binning 对象；若为 None，则用训练集重新拟合
        n_bins      : binning 为 None 时使用的分箱数
        filename    : 输出文件名（不含扩展名）
        """
        from .binning import Binning

        # 用训练集确定箱边界
        if binning is None:
            train_df = df[df[dataset_col] == "train"].reset_index(drop=True)
            if train_df.empty:
                train_df = df.reset_index(drop=True)
            binning = Binning(n_bins=n_bins)
            binning.fit(train_df[features + [target]], target=target, cols=features)

        datasets = sorted(df[dataset_col].dropna().unique())
        excel_path = os.path.join(self.output_dir, f"{filename}.xlsx")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for feat in features:
                if feat not in binning._bin_edges:
                    continue
                edges = binning._bin_edges[feat]
                all_rows = []
                for ds in datasets:
                    sub = df[df[dataset_col] == ds][[feat, target]].copy()
                    if sub.empty:
                        continue
                    sub["_bin"] = pd.cut(sub[feat], bins=edges, include_lowest=True)
                    total_n = len(sub)
                    total_bad = int(sub[target].sum())
                    grp_rows = []
                    for bin_label, g in sub.groupby("_bin", observed=True):
                        n   = len(g)
                        bad = int(g[target].sum())
                        grp_rows.append({
                            "dataset": ds,
                            "分箱区间": str(bin_label),
                            "样本数": n,
                            "样本占比": round(n / total_n, 4) if total_n > 0 else None,
                            "坏样本数": bad,
                            "坏样本率": round(bad / n, 4) if n > 0 else None,
                            "坏样本在全集占比": round(bad / total_bad, 4) if total_bad > 0 else None,
                        })
                    all_rows.extend(grp_rows)

                if all_rows:
                    feat_df = pd.DataFrame(all_rows)
                    sheet = feat[:31]
                    feat_df.to_excel(writer, sheet_name=sheet, index=False)

        return excel_path

    # ------------------------------------------------------------------ #
    #  内部辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _df_to_html_table(df: pd.DataFrame) -> str:
        return df.to_html(index=False, border=1, classes="tbl", justify="center")

    def _build_clf_html(self, summary_df, sheets, title) -> str:
        css = ("<style>body{font-family:微软雅黑,Arial;margin:20px}"
               "h1,h2{color:#2c3e50}.tbl{border-collapse:collapse;width:100%;margin-bottom:20px}"
               ".tbl th{background:#2c3e50;color:#fff;padding:6px 10px}"
               ".tbl td{padding:5px 10px;border:1px solid #ccc}"
               ".tbl tr:nth-child(even){background:#f2f2f2}</style>")
        body = f"<h1>{title}</h1><h2>汇总指标</h2>" + self._df_to_html_table(summary_df)
        for name, df in sheets.items():
            body += f"<h2>{name}</h2>" + self._df_to_html_table(df)
        return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>{title}</title>{css}</head><body>{body}</body></html>")

    def _build_reg_html(self, summary_df, sheets, title) -> str:
        css = ("<style>body{font-family:微软雅黑,Arial;margin:20px}"
               "h1,h2{color:#2c3e50}.tbl{border-collapse:collapse;width:100%;margin-bottom:20px}"
               ".tbl th{background:#2c3e50;color:#fff;padding:6px 10px}"
               ".tbl td{padding:5px 10px;border:1px solid #ccc}"
               ".tbl tr:nth-child(even){background:#f2f2f2}</style>")
        body = f"<h1>{title}</h1><h2>汇总指标</h2>" + self._df_to_html_table(summary_df)
        for name, df in sheets.items():
            body += f"<h2>{name}</h2>" + self._df_to_html_table(df)
        return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>{title}</title>{css}</head><body>{body}</body></html>")


# ------------------------------------------------------------------ #
#  模块级工具函数
# ------------------------------------------------------------------ #
def _detect_month_col(df: pd.DataFrame) -> Optional[str]:
    """自动检测月份列名（优先 'month'，其次 'ym'，再次含 'month'/'ym' 的列）"""
    for cand in ["month", "ym", "月份", "yearmonth", "year_month"]:
        if cand in df.columns:
            return cand
    for col in df.columns:
        if "month" in col.lower() or "ym" in col.lower():
            return col
    return None
