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


def _label_rate(labels) -> Optional[float]:
    """计算二值列中 1 的占比：1值数量 / (0值数量 + 1值数量)，过滤掉非 0/1 的值。"""
    arr = np.array(labels)
    mask = (arr == 0) | (arr == 1)
    valid = arr[mask]
    if len(valid) == 0:
        return None
    return round(float(valid.sum()) / len(valid), 4)


def _build_feature_importance_df(feature_importance) -> Optional[pd.DataFrame]:
    """
    把传入的特征重要性数据整理成标准 DataFrame，供写入 Excel sheet。
    接受以下格式：
      - None / 空 → 返回 None
      - pd.DataFrame（须含 'feature' 和 'importance' 列）
      - dict {feature: importance}
      - list of (feature, importance) 元组
    返回按重要性降序排列的 DataFrame，含以下列：
      排名 | 特征名 | 重要性得分 | 占比(%) | 累计占比(%)
    """
    if feature_importance is None:
        return None

    # 统一转成 DataFrame
    if isinstance(feature_importance, dict):
        fi = pd.DataFrame(list(feature_importance.items()), columns=["feature", "importance"])
    elif isinstance(feature_importance, (list, tuple)):
        fi = pd.DataFrame(feature_importance, columns=["feature", "importance"])
    elif isinstance(feature_importance, pd.DataFrame):
        fi = feature_importance.copy()
    else:
        return None

    if fi.empty or "feature" not in fi.columns or "importance" not in fi.columns:
        return None

    fi = fi[["feature", "importance"]].copy()
    fi["importance"] = pd.to_numeric(fi["importance"], errors="coerce").fillna(0)
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    fi.index += 1

    total = fi["importance"].sum()
    fi["占比(%)"]   = (fi["importance"] / total * 100).round(2) if total > 0 else 0.0
    fi["累计占比(%)"] = fi["占比(%)"].cumsum().round(2)

    fi = fi.reset_index().rename(columns={"index": "排名", "feature": "特征名", "importance": "重要性得分"})
    return fi[["排名", "特征名", "重要性得分", "占比(%)", "累计占比(%)"]]


def _make_bin_labels(mins: list, maxs: list) -> list:
    """根据每箱的实际最小值/最大值生成首尾相连的区间标签。
    相邻箱边界对齐：第 i 箱右端 = 第 i+1 箱左端（取两者边界的均值）。
    首箱左端 = 首箱最小值，末箱右端 = 末箱最大值。
    """
    n = len(mins)
    lefts  = [None] * n
    rights = [None] * n
    lefts[0]    = mins[0]
    rights[-1]  = maxs[-1]
    for i in range(n - 1):
        mid = round((maxs[i] + mins[i + 1]) / 2, 6)
        rights[i]     = mid
        lefts[i + 1]  = mid
    labels = []
    for i, (l, r) in enumerate(zip(lefts, rights)):
        lp = "(" if i > 0 else "["
        labels.append(f"{lp}{l}, {r}]")
    return labels


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
        return pd.DataFrame(columns=["分箱", "分箱区间", "样本数", "样本占比", "坏样本数",
                                     "加权样本数", "最低分", "最高分", "坏样本率", "Lift"])
    grp = pd.DataFrame(rows)
    grp.insert(1, "分箱区间", _make_bin_labels(
        grp["最低分"].tolist(), grp["最高分"].tolist()))
    return grp


def _bucket_table(y_true, y_pred, n_bins: int = 10, bins: list = None,
                  by: str = "true", cut_edges: list = None,
                  labels: np.ndarray = None) -> pd.DataFrame:
    """按真实值或预测值分桶，统计每桶误差指标。

    by:        'true' — 按真实值分桶；'pred' — 按预测值分桶
    bins:      自定义切点，传入时忽略 n_bins，自动补 ±inf
    cut_edges: 已计算好的完整 bin 边界（含 ±inf），优先级高于 bins/n_bins
    labels:    与 y_true 等长的二值数组（0/1），传入时在分桶结果末尾加 '1值占比' 列
    """
    total_n = len(y_true)
    df = pd.DataFrame({"真实值": np.array(y_true), "预测值": np.array(y_pred)})
    if labels is not None:
        df["_label"] = np.array(labels)
    col = "真实值" if by == "true" else "预测值"
    try:
        if cut_edges is not None:
            df["分桶"] = pd.cut(df[col], bins=cut_edges, labels=False, include_lowest=True)
            df["分桶"] = df["分桶"] + 1
        elif bins is not None:
            edges = ([-np.inf] + list(bins) + [np.inf]
                     if (bins[0] != -np.inf and bins[-1] != np.inf) else bins)
            df["分桶"] = pd.cut(df[col], bins=edges, labels=False, include_lowest=True)
            df["分桶"] = df["分桶"] + 1
        else:
            df["分桶"] = pd.qcut(df[col], q=n_bins, duplicates="drop", labels=False) + 1
    except ValueError:
        df["分桶"] = 1

    rows = []
    for bin_id, g in df.groupby("分桶"):
        n = len(g)
        mae_val  = round(float(mean_absolute_error(g["真实值"].values, g["预测值"].values)), 4)
        mape_val = _mape(g["真实值"].values, g["预测值"].values)
        row = {
            "分桶": int(bin_id),
            "样本数": n,
            "样本占比": round(n / total_n, 4),
            "真实最小值": round(float(g["真实值"].min()), 4),
            "真实最大值": round(float(g["真实值"].max()), 4),
            "真实均值": round(float(g["真实值"].mean()), 4),
            "预测均值": round(float(g["预测值"].mean()), 4),
            "预测最低值": round(float(g["预测值"].min()), 4),
            "预测最高值": round(float(g["预测值"].max()), 4),
            "MAE": mae_val,
            "MAPE(%)": mape_val,
        }
        if labels is not None:
            row["1值占比"] = _label_rate(g["_label"].values)
        rows.append(row)

    base_cols = ["分桶", "分箱区间", "样本数", "样本占比", "真实最小值", "真实最大值",
                 "真实均值", "预测均值", "预测最低值", "预测最高值",
                 "MAE", "MAPE(%)", "误差(预测-真实)"]
    if labels is not None:
        base_cols.append("1值占比")
    if not rows:
        return pd.DataFrame(columns=base_cols)
    grp = pd.DataFrame(rows)
    grp["误差(预测-真实)"] = (grp["预测均值"] - grp["真实均值"]).round(4)
    grp.insert(1, "分箱区间", _make_bin_labels(
        grp["真实最小值"].tolist() if by == "true" else grp["预测最低值"].tolist(),
        grp["真实最大值"].tolist() if by == "true" else grp["预测最高值"].tolist()))
    return grp


def _build_cross_matrix(y_true, y_pred, true_edges: list, pred_edges: list,
                        labels: np.ndarray = None):
    """计算真实值桶 × 预测值桶交叉矩阵，返回 (mape_df, count_df[, label_df]) 。

    mape_df:  格子值为 MAPE(%)
    count_df: 格子值为 "样本数(占比%)"
    label_df: 格子值为 1值占比（仅 labels 传入时返回，否则该位置为 None）
    """
    total_n = len(y_true)
    df = pd.DataFrame({"真实值": np.array(y_true), "预测值": np.array(y_pred)})
    if labels is not None:
        df["_label"] = np.array(labels)
    df["真实桶"] = pd.cut(df["真实值"], bins=true_edges, labels=False, include_lowest=True).astype("Int64") + 1
    df["预测桶"] = pd.cut(df["预测值"], bins=pred_edges, labels=False, include_lowest=True).astype("Int64") + 1

    true_bins = sorted(df["真实桶"].dropna().unique())
    pred_bins = sorted(df["预测桶"].dropna().unique())
    col_label = "真实值桶\\预测值桶"
    pred_cols  = [f"预测桶{int(pb)}" for pb in pred_bins]

    mape_rows, count_rows, label_rows = [], [], []
    for tb in true_bins:
        sub_t  = df[df["真实桶"] == tb]
        mrow   = {col_label: f"真实桶{int(tb)}"}
        crow   = {col_label: f"真实桶{int(tb)}"}
        lrow   = {col_label: f"真实桶{int(tb)}"}
        for pb, pc in zip(pred_bins, pred_cols):
            sub = sub_t[sub_t["预测桶"] == pb]
            n   = len(sub)
            mrow[pc] = round(_mape(sub["真实值"].values, sub["预测值"].values), 2) if n else None
            crow[pc] = f"{n}({round(100*n/total_n,1)}%)" if n else "0(0.0%)"
            if labels is not None:
                lrow[pc] = _label_rate(sub["_label"].values)
        rt = len(sub_t)
        mrow["合计"] = round(_mape(sub_t["真实值"].values, sub_t["预测值"].values), 2) if rt else None
        crow["合计"] = f"{rt}({round(100*rt/total_n,1)}%)"
        if labels is not None:
            lrow["合计"] = _label_rate(sub_t["_label"].values)
        mape_rows.append(mrow)
        count_rows.append(crow)
        if labels is not None:
            label_rows.append(lrow)

    # 列合计行
    mrow_total  = {col_label: "合计"}
    crow_total  = {col_label: "合计"}
    lrow_total  = {col_label: "合计"}
    for pb, pc in zip(pred_bins, pred_cols):
        sub_p = df[df["预测桶"] == pb]
        np_   = len(sub_p)
        mrow_total[pc] = round(_mape(sub_p["真实值"].values, sub_p["预测值"].values), 2) if np_ else None
        crow_total[pc] = f"{np_}({round(100*np_/total_n,1)}%)"
        if labels is not None:
            lrow_total[pc] = _label_rate(sub_p["_label"].values)
    mrow_total["合计"] = round(_mape(df["真实值"].values, df["预测值"].values), 2)
    crow_total["合计"] = f"{total_n}(100.0%)"
    if labels is not None:
        lrow_total["合计"] = _label_rate(df["_label"].values)
    mape_rows.append(mrow_total)
    count_rows.append(crow_total)
    if labels is not None:
        label_rows.append(lrow_total)

    label_df = pd.DataFrame(label_rows) if labels is not None else None
    return pd.DataFrame(mape_rows), pd.DataFrame(count_rows), label_df


def _get_cut_edges(values, n_bins: int, bins: list) -> list:
    """根据给定数组计算分桶切点边界（含 ±inf），供多集合共享。
    bins 传入时直接转为边界，否则对 values 做等频 qcut。
    """
    if bins is not None:
        return ([-np.inf] + list(bins) + [np.inf]
                if (bins[0] != -np.inf and bins[-1] != np.inf) else list(bins))
    try:
        _, edges = pd.qcut(np.array(values), q=n_bins, duplicates="drop", retbins=True)
        edges[0]  = -np.inf
        edges[-1] = np.inf
        return list(edges)
    except ValueError:
        return [-np.inf, np.inf]


class ReportGenerator:
    """生成二分类 / 回归模型评估报告，以及特征分箱报告，输出 Excel 和 HTML

    目录结构（自动创建）：
        output_dir/
        ├── 01_feature_analysis/   特征分析报告、分箱报告
        ├── 02_feature_selection/  特征筛选报告
        ├── 03_model_tuning/       调参日志（由 ModelTrainer.fit 写入）
        ├── 04_model_report/       模型评估报告（Excel + HTML）
        └── 05_model_deploy/       模型部署文件（由 ModelTrainer.save 写入）
    """

    SUBDIRS = [
        "01_feature_analysis",
        "02_feature_selection",
        "03_model_tuning",
        "04_model_report",
        "05_model_deploy",
    ]

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        for sub in self.SUBDIRS:
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    def _path(self, subdir: str, filename: str) -> str:
        return os.path.join(self.output_dir, subdir, filename)

    # ------------------------------------------------------------------ #
    #  二分类报告
    # ------------------------------------------------------------------ #
    def classification_report(
        self,
        datasets: dict,
        filename: str = "分类模型评估报告",
        month_col_data: Optional[dict] = None,
        feature_importance: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        datasets:           {"数据集名": (y_true, y_score[, weight]), ...}
        month_col_data:     {"数据集名": df_with_month_col}，若为 None 则不生成按月 sheet
        feature_importance: 含 'feature' 和 'importance' 列的 DataFrame，
                            传入后自动生成「特征重要性」sheet（按重要性降序，含归一化占比和累计占比）
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
        excel_path = self._path("04_model_report", f"{filename}.xlsx")
        html_path  = self._path("04_model_report", f"{filename}.html")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="汇总指标", index=False)
            for sname, sdf in sheets.items():
                sdf.to_excel(writer, sheet_name=sname[:31], index=False)
            fi_df = _build_feature_importance_df(feature_importance)
            if fi_df is not None:
                fi_df.to_excel(writer, sheet_name="特征重要性", index=False)

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
        bins: list = None,
        n_bins_pred: int = 10,
        bins_pred: list = None,
        month_col_data: Optional[dict] = None,
        label_col_data: Optional[dict] = None,
        feature_importance: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        datasets:       {"数据集名": (y_true, y_pred), ...}
                        第一个数据集视为训练集，用于计算共享切点。
        n_bins:         真实值等频分桶数（bins 未传时生效），默认 10
        bins:           真实值自定义切点（传入时忽略 n_bins）
        n_bins_pred:    预测值等频分桶数（bins_pred 未传时生效），默认 10
        bins_pred:      预测值自定义切点（传入时忽略 n_bins_pred）
        label_col_data: {"数据集名": array_like}，各集合对应的二值列（0/1），
                        传入后分桶表末尾加 '1值占比'，矩阵增加 1值占比矩阵
        真实值分桶：切点由训练集真实值计算，三集合共享，合并为一张 sheet "真实值分桶"
        预测值分桶：切点由训练集预测值计算，三集合共享，合并为一张 sheet "预测值分桶"
        分桶矩阵：三集合合并为一张 sheet "分桶矩阵"，含 MAPE / 样本数占比 / 1值占比（可选）矩阵
        """
        summary_rows = []
        sheets = {}

        # 用第一个数据集（训练集）计算两套共享切点
        first_vals = list(datasets.values())[0]
        true_edges = _get_cut_edges(np.array(first_vals[0]), n_bins,      bins)
        pred_edges = _get_cut_edges(np.array(first_vals[1]), n_bins_pred, bins_pred)

        merged_true_rows = []
        merged_pred_rows = []
        merged_matrix_rows = []
        for name, (y_true, y_pred) in datasets.items():
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            lbl = np.array(label_col_data[name]) if label_col_data and name in label_col_data else None

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
            bdf_t = _bucket_table(y_true, y_pred, by="true", cut_edges=true_edges, labels=lbl)
            bdf_t.insert(0, "数据集", name)
            merged_true_rows.append(bdf_t)

            bdf_p = _bucket_table(y_true, y_pred, by="pred", cut_edges=pred_edges, labels=lbl)
            bdf_p.insert(0, "数据集", name)
            merged_pred_rows.append(bdf_p)

            merged_matrix_rows.append((name, _build_cross_matrix(
                y_true, y_pred, true_edges, pred_edges, labels=lbl)))

        # 三张 sheet：真实值分桶、预测值分桶、分桶矩阵，各自合并三集合
        def _section_header(title: str, cols: list) -> pd.DataFrame:
            row = {c: "" for c in cols}
            row[cols[0]] = f"── {title} ──"
            return pd.DataFrame([row])

        def _blank_row(cols: list) -> pd.DataFrame:
            return pd.DataFrame([{c: "" for c in cols}])

        if merged_true_rows:
            sheets["真实值分桶"] = pd.concat(merged_true_rows, ignore_index=True)
        if merged_pred_rows:
            sheets["预测值分桶"] = pd.concat(merged_pred_rows, ignore_index=True)

        # 分桶矩阵 sheet：三集合合并，含 MAPE + 样本数/占比 + 1值占比（可选）
        mat_parts = []
        for ds_name, (mape_df, count_df, label_df) in merged_matrix_rows:
            if mat_parts:
                mat_parts.append(_blank_row(mape_df.columns.tolist()))
            mat_parts.append(_section_header(f"{ds_name} — MAPE矩阵（行=真实值桶，列=预测值桶）", mape_df.columns.tolist()))
            mat_parts.append(mape_df)
            mat_parts.append(_blank_row(count_df.columns.tolist()))
            mat_parts.append(_section_header(f"{ds_name} — 样本数/占比矩阵", count_df.columns.tolist()))
            mat_parts.append(count_df)
            if label_df is not None:
                mat_parts.append(_blank_row(label_df.columns.tolist()))
                mat_parts.append(_section_header(f"{ds_name} — 1值占比矩阵", label_df.columns.tolist()))
                mat_parts.append(label_df)
        if mat_parts:
            sheets["分桶矩阵"] = pd.concat(mat_parts, ignore_index=True)

        # 按月分桶 sheet
        if month_col_data:
            for name, mdf in month_col_data.items():
                if name not in datasets:
                    continue
                y_true_all, y_pred_all = np.array(datasets[name][0]), np.array(datasets[name][1])
                lbl_all = np.array(label_col_data[name]) if label_col_data and name in label_col_data else None
                mdf = mdf.reset_index(drop=True)
                month_col = _detect_month_col(mdf)
                if month_col is None:
                    continue
                tmp = pd.DataFrame({"y": y_true_all, "p": y_pred_all,
                                    "month": mdf[month_col].values})
                if lbl_all is not None:
                    tmp["_label"] = lbl_all
                rows_true, rows_pred = [], []
                for month_val, g in tmp.groupby("month", sort=True):
                    lbl_g = g["_label"].values if lbl_all is not None else None
                    bdf_t = _bucket_table(g["y"].values, g["p"].values,
                                          by="true", cut_edges=true_edges, labels=lbl_g)
                    bdf_p = _bucket_table(g["y"].values, g["p"].values,
                                          by="pred", cut_edges=pred_edges, labels=lbl_g)
                    bdf_t.insert(0, month_col, month_val)
                    bdf_p.insert(0, month_col, month_val)
                    rows_true.append(bdf_t)
                    rows_pred.append(bdf_p)
                if rows_true:
                    sheets[f"{name}_按月真实值分桶"[:31]] = pd.concat(rows_true, ignore_index=True)
                if rows_pred:
                    sheets[f"{name}_按月预测值分桶"[:31]] = pd.concat(rows_pred, ignore_index=True)

        summary_df = pd.DataFrame(summary_rows)
        excel_path = self._path("04_model_report", f"{filename}.xlsx")
        html_path  = self._path("04_model_report", f"{filename}.html")

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="汇总指标", index=False)
            for sname, sdf in sheets.items():
                sdf.to_excel(writer, sheet_name=sname[:31], index=False)
            fi_df = _build_feature_importance_df(feature_importance)
            if fi_df is not None:
                fi_df.to_excel(writer, sheet_name="特征重要性", index=False)

        html = self._build_reg_html(summary_df, sheets, filename)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return {"summary": summary_df, "sheets": sheets,
                "excel_path": excel_path, "html_path": html_path}

    # ------------------------------------------------------------------ #
    #  特征分析报告（整体 + by dataset + by month，汇总到一个 Excel）
    # ------------------------------------------------------------------ #
    def feature_analysis_report(
        self,
        df: pd.DataFrame,
        features: list,
        dataset_col: Optional[str] = "dataset",
        month_col: Optional[str] = "month",
        cat_cols: Optional[list] = None,
        filename: str = "特征分析报告",
    ) -> str:
        """
        将整体、by dataset、by month 三个维度的缺失率 / 一值率 / 分位数统计
        汇总输出到同一个 Excel，共 3 个 sheet：
          - 整体        : 缺失率 → 空行 → 一值率 → 空行 → 分位数统计
          - by_dataset  : 同上，各指标前有分组列
          - by_month    : 同上，各指标前有分组列
        """
        from .feature_analysis import FeatureAnalyzer

        cat_cols = cat_cols or []
        analyzer = FeatureAnalyzer(df[features], cat_cols=cat_cols)

        def _stack(blocks: list) -> pd.DataFrame:
            """把多个 DataFrame 纵向拼接，中间插入标题行 + 空行。
            blocks: [(title, df), ...]
            """
            parts = []
            for title, bdf in blocks:
                if bdf is None or (isinstance(bdf, pd.DataFrame) and bdf.empty):
                    continue
                # 标题行（单列，其余 NaN）
                header = pd.DataFrame([[title] + [np.nan] * (len(bdf.columns) - 1)],
                                      columns=bdf.columns)
                parts.append(header)
                parts.append(bdf)
                # 空行
                parts.append(pd.DataFrame([[np.nan] * len(bdf.columns)], columns=bdf.columns))
            return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        excel_path = self._path("01_feature_analysis", f"{filename}.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

            # ── 整体 ──────────────────────────────────────────────────
            mr = analyzer.missing_rate()
            sr = analyzer.single_value_rate()
            qs = analyzer.quantile_stats()
            overall = _stack([("【缺失率】", mr), ("【一值率】", sr), ("【分位数统计】", qs)])
            if not overall.empty:
                overall.to_excel(writer, sheet_name="整体", index=False)

            # ── by dataset ────────────────────────────────────────────
            if dataset_col and dataset_col in df.columns:
                raw = df[features + [dataset_col]].copy()
                by_ds = analyzer.by_group_analysis(dataset_col, raw)
                ds_sheet = _stack([
                    ("【缺失率】",     by_ds.get("缺失率")),
                    ("【一值率】",     by_ds.get("一值率")),
                    ("【分位数统计】", by_ds.get("分位数统计")),
                ])
                if not ds_sheet.empty:
                    ds_sheet.to_excel(writer, sheet_name="by_dataset", index=False)

            # ── by month ──────────────────────────────────────────────
            if month_col and month_col in df.columns:
                raw_m = df[features + [month_col]].copy()
                by_m = analyzer.by_group_analysis(month_col, raw_m)
                m_sheet = _stack([
                    ("【缺失率】",     by_m.get("缺失率")),
                    ("【一值率】",     by_m.get("一值率")),
                    ("【分位数统计】", by_m.get("分位数统计")),
                ])
                if not m_sheet.empty:
                    m_sheet.to_excel(writer, sheet_name="by_month", index=False)

        return excel_path

    # ------------------------------------------------------------------ #
    #  特征筛选 Excel 报告
    # ------------------------------------------------------------------ #
    def feature_selection_report(
        self,
        feature_report_df: pd.DataFrame,
        analysis_results: Optional[dict] = None,
        filename: str = "特征筛选报告",
    ) -> str:
        excel_path = self._path("02_feature_selection", f"{filename}.xlsx")
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
        excel_path = self._path("01_feature_analysis", f"{filename}.xlsx")

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
                    for bin_label, g in sub.groupby("_bin"):
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
