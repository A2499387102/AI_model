import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Optional
from sklearn.metrics import roc_auc_score, roc_curve, mean_squared_error


def _ks_auc(y_true, y_score, weights=None):
    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=weights)
    auc = roc_auc_score(y_true, y_score, sample_weight=weights)
    if auc < 0.5:
        auc = 1 - auc
    ks = round(float(np.abs(tpr - fpr).max()), 4)
    return ks, round(float(auc), 4)


class LGBMTrainer:
    """LightGBM 二分类训练器"""

    def __init__(
        self,
        n_trials: int = 150,
        use_gpu: bool = False,
        cat_cols: Optional[list] = None,
        random_state: int = 2024,
    ):
        self.n_trials = n_trials
        self.use_gpu = use_gpu
        self.cat_cols = cat_cols or []
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.selected_features: list = []
        self.trials_log: pd.DataFrame = pd.DataFrame()

    def get_default_space(self, n_data: int) -> dict:
        """
        返回默认超参数搜索空间字典。

        在 Jupyter 中不重启修改参数的用法：
            space = trainer.get_default_space(len(X_train))
            # 修改想调整的参数范围，例如：
            from hyperopt import hp
            space['learning_rate'] = hp.uniform('learning_rate', 0.001, 0.05)
            space['max_depth'] = hp.quniform('max_depth', 3, 6, 1)
            # 传入 fit
            trainer.fit(..., custom_space=space)
        """
        from hyperopt import hp
        return {
            "boosting_type": "gbdt", "objective": "binary", "metric": "auc",
            "is_unbalance": True, "feature_pre_filter": True, "nthread": -1, "verbose": -1,
            "bagging_freq": 2,
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 0.8, 0.1),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 0.8, 0.1),
            "max_depth": hp.quniform("max_depth", 2, 4, 1),
            "num_leaves": hp.quniform("num_leaves", 2, 50, 2),
            "n_estimators": hp.quniform("n_estimators", 300, 500, 10),
            "learning_rate": hp.uniform("learning_rate", 0.005, 0.02),
            "lambda_l1": hp.randint("lambda_l1", 10, 50),
            "lambda_l2": hp.randint("lambda_l2", 10, 50),
            "min_data_in_leaf": hp.quniform(
                "min_data_in_leaf",
                max(int(n_data * 0.01), 1),
                max(int(n_data * 0.05), 2),
                100,
            ),
        }

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_oot: pd.DataFrame,
        y_oot: pd.Series,
        train_weight: Optional[pd.Series] = None,
        test_weight: Optional[pd.Series] = None,
        oot_weight: Optional[pd.Series] = None,
        features: Optional[list] = None,
        custom_space: Optional[dict] = None,
        save_dir: Optional[str] = None,
    ) -> "LGBMTrainer":
        try:
            import lightgbm as lgb
            from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
        except ImportError as e:
            raise ImportError(f"请安装依赖: pip install lightgbm hyperopt — {e}")

        features = features or X_train.columns.tolist()
        Xtr  = X_train[features].reset_index(drop=True)
        Xts  = X_test[features].reset_index(drop=True)
        Xoot = X_oot[features].reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test  = y_test.reset_index(drop=True)
        y_oot   = y_oot.reset_index(drop=True)
        tw = train_weight.reset_index(drop=True) if train_weight is not None else pd.Series(np.ones(len(Xtr)))
        vw = test_weight.reset_index(drop=True)  if test_weight  is not None else pd.Series(np.ones(len(Xts)))
        ow = oot_weight.reset_index(drop=True)   if oot_weight   is not None else pd.Series(np.ones(len(Xoot)))

        train_set = lgb.Dataset(Xtr, y_train, categorical_feature=self.cat_cols, weight=tw, free_raw_data=False)
        test_set  = lgb.Dataset(Xts, y_test,  categorical_feature=self.cat_cols, weight=vw, free_raw_data=False)

        logs = []

        def objective(params):
            params["num_leaves"]      = int(params["num_leaves"])
            params["max_depth"]       = int(params["max_depth"])
            num_boost_round           = int(params.pop("n_estimators"))
            params["lambda_l1"]       = int(params["lambda_l1"])
            params["lambda_l2"]       = int(params["lambda_l2"])
            params["min_data_in_leaf"]= int(params["min_data_in_leaf"])
            m = lgb.train(params, train_set, num_boost_round=num_boost_round, valid_sets=test_set,
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            tr_auc  = roc_auc_score(y_train, m.predict(Xtr,  num_iteration=m.best_iteration), sample_weight=tw)
            ts_auc  = roc_auc_score(y_test,  m.predict(Xts,  num_iteration=m.best_iteration), sample_weight=vw)
            oot_auc = roc_auc_score(y_oot,   m.predict(Xoot, num_iteration=m.best_iteration), sample_weight=ow)
            loss = -oot_auc + abs(tr_auc - ts_auc)
            logs.append({"train_auc": round(tr_auc, 4), "test_auc": round(ts_auc, 4),
                         "oot_auc": round(oot_auc, 4), "loss": round(loss, 4),
                         "params": {**params, "n_estimators": num_boost_round}})
            return {"loss": loss, "status": STATUS_OK}

        n_data = len(Xtr)
        space  = custom_space if custom_space is not None else self.get_default_space(n_data)
        trials = Trials()
        fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=self.n_trials, trials=trials, verbose=False)
        self.trials_log = pd.DataFrame(logs)

        best_row = self.trials_log.loc[self.trials_log["loss"].idxmin(), "params"]
        params1  = dict(best_row)
        num_br1  = int(params1.pop("n_estimators", 300))
        model1   = lgb.train(params1, train_set, num_boost_round=num_br1, valid_sets=test_set,
                             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

        imp = pd.DataFrame({"feature": model1.feature_name(), "importance": model1.feature_importance()})
        self.selected_features = imp[imp["importance"] > 0]["feature"].tolist()
        if not self.selected_features:
            self.selected_features = features

        Xtr2, Xts2, Xoot2 = Xtr[self.selected_features], Xts[self.selected_features], Xoot[self.selected_features]
        train_set2 = lgb.Dataset(Xtr2, y_train, categorical_feature=self.cat_cols, weight=tw, free_raw_data=False)
        test_set2  = lgb.Dataset(Xts2, y_test,  categorical_feature=self.cat_cols, weight=vw, free_raw_data=False)

        logs2 = []

        def objective2(params):
            params["num_leaves"]      = int(params["num_leaves"])
            params["max_depth"]       = int(params["max_depth"])
            num_boost_round2          = int(params.pop("n_estimators"))
            params["lambda_l1"]       = int(params["lambda_l1"])
            params["lambda_l2"]       = int(params["lambda_l2"])
            params["min_data_in_leaf"]= int(params["min_data_in_leaf"])
            m = lgb.train(params, train_set2, num_boost_round=num_boost_round2, valid_sets=test_set2,
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            tr_auc  = roc_auc_score(y_train, m.predict(Xtr2,  num_iteration=m.best_iteration), sample_weight=tw)
            ts_auc  = roc_auc_score(y_test,  m.predict(Xts2,  num_iteration=m.best_iteration), sample_weight=vw)
            oot_auc = roc_auc_score(y_oot,   m.predict(Xoot2, num_iteration=m.best_iteration), sample_weight=ow)
            loss = -oot_auc + abs(tr_auc - ts_auc)
            logs2.append({"train_auc": round(tr_auc, 4), "test_auc": round(ts_auc, 4),
                          "oot_auc": round(oot_auc, 4), "loss": round(loss, 4)})
            return {"loss": loss, "status": STATUS_OK}

        space2 = dict(space)
        space2["feature_pre_filter"] = False
        space2["bagging_fraction"]   = hp.quniform("bagging_fraction", 0.6, 0.8, 0.1)
        space2["feature_fraction"]   = hp.quniform("feature_fraction", 0.6, 0.8, 0.1)
        space2["n_estimators"]       = hp.quniform("n_estimators", 200, 400, 10)
        trials2 = Trials()
        fmin(fn=objective2, space=space2, algo=tpe.suggest, max_evals=self.n_trials, trials=trials2, verbose=False)

        idx2    = int(np.argmin([r["loss"] for r in logs2]))
        params2 = trials2.trials[idx2]["misc"]["vals"]
        params2 = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in params2.items()}
        num_br_final = int(params2.pop("n_estimators", 300))
        params2.update({"boosting_type": "gbdt", "objective": "binary", "metric": "auc",
                        "is_unbalance": True, "feature_pre_filter": False,
                        "nthread": -1, "verbose": -1, "bagging_freq": 2})
        for k in ["num_leaves", "max_depth", "lambda_l1", "lambda_l2", "min_data_in_leaf"]:
            if k in params2:
                params2[k] = int(params2[k])
        self.model = lgb.train(params2, train_set2, num_boost_round=num_br_final, valid_sets=test_set2,
                               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        self.best_params = {**params2, "n_estimators": num_br_final}

        if save_dir:
            self.save(save_dir)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        feats = self.selected_features or X.columns.tolist()
        return self.model.predict(X[feats], num_iteration=self.model.best_iteration)

    def save(self, save_dir: str) -> None:
        """保存模型文件和入模特征列表到 save_dir"""
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_model(os.path.join(save_dir, "lgbm_model.txt"))
        with open(os.path.join(save_dir, "selected_features.json"), "w", encoding="utf-8") as f:
            json.dump(self.selected_features, f, ensure_ascii=False, indent=2)
        meta = {"model_type": "lgbm", "best_params": self.best_params,
                "selected_features": self.selected_features}
        with open(os.path.join(save_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"模型已保存至: {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> "LGBMTrainer":
        """从 save_dir 恢复模型"""
        import lightgbm as lgb
        trainer = cls()
        trainer.model = lgb.Booster(model_file=os.path.join(save_dir, "lgbm_model.txt"))
        with open(os.path.join(save_dir, "selected_features.json"), encoding="utf-8") as f:
            trainer.selected_features = json.load(f)
        with open(os.path.join(save_dir, "model_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        trainer.best_params = meta.get("best_params", {})
        return trainer


class XGBTrainer:
    """XGBoost 二分类训练器"""

    def __init__(self, n_trials: int = 100, use_gpu: bool = False, random_state: int = 2024):
        self.n_trials = n_trials
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.selected_features: list = []
        self.trials_log: pd.DataFrame = pd.DataFrame()

    def get_default_space(self, odds: float) -> dict:
        """
        返回默认超参数搜索空间字典。

        在 Jupyter 中不重启修改参数的用法：
            space = trainer.get_default_space(odds)
            from hyperopt import hp
            space['learning_rate'] = hp.quniform('learning_rate', 0.005, 0.1, 0.005)
            trainer.fit(..., custom_space=space)
        """
        from hyperopt import hp
        return {
            "base_score": 0.5, "booster": "gbtree", "objective": "binary:logistic",
            "learning_rate": hp.quniform("learning_rate", 0.01, 0.3, 0.01),
            "gamma": hp.quniform("gamma", 0, 200, 2),
            "max_depth": hp.choice("max_depth", [2, 3]),
            "subsample": hp.quniform("subsample", 0.5, 1, 0.1),
            "colsample_bytree": 1,
            "n_estimators": 500,
            "min_child_weight": hp.quniform("min_child_weight", 100, 1000, 50),
            "reg_alpha": hp.quniform("reg_alpha", 0, 300, 10),
            "reg_lambda": hp.quniform("reg_lambda", 0, 300, 10),
            "scale_pos_weight": hp.quniform("scale_pos_weight", max(int(odds) - 2, 1), int(odds) + 2, 1),
            "random_state": self.random_state,
            "tree_method": "gpu_hist" if self.use_gpu else "hist",
            "early_stopping_rounds": 30,
            "eval_metric": "auc",
            "nthread": -1,
        }

    def fit(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test:  pd.DataFrame, y_test:  pd.Series,
        X_oot:   pd.DataFrame, y_oot:   pd.Series,
        train_weight: Optional[pd.Series] = None,
        test_weight:  Optional[pd.Series] = None,
        oot_weight:   Optional[pd.Series] = None,
        features: Optional[list] = None,
        custom_space: Optional[dict] = None,
        save_dir: Optional[str] = None,
    ) -> "XGBTrainer":
        try:
            import xgboost as xgb
            from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
        except ImportError as e:
            raise ImportError(f"请安装依赖: pip install xgboost hyperopt — {e}")

        features = features or X_train.columns.tolist()
        Xtr  = X_train[features].reset_index(drop=True)
        Xts  = X_test[features].reset_index(drop=True)
        Xoot = X_oot[features].reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test  = y_test.reset_index(drop=True)
        y_oot   = y_oot.reset_index(drop=True)
        tw = train_weight.reset_index(drop=True) if train_weight is not None else pd.Series(np.ones(len(Xtr)))
        vw = test_weight.reset_index(drop=True)  if test_weight  is not None else pd.Series(np.ones(len(Xts)))
        ow = oot_weight.reset_index(drop=True)   if oot_weight   is not None else pd.Series(np.ones(len(Xoot)))

        tree_method = "gpu_hist" if self.use_gpu else "hist"
        odds = max(float((tw[y_train == 0].sum()) / (tw[y_train == 1].sum() + 1e-6)), 1.0)
        logs = []

        def objective(params):
            clf = xgb.XGBClassifier(**params, verbosity=0)
            clf.fit(Xtr, y_train, sample_weight=tw,
                    eval_set=[(Xts, y_test)], sample_weight_eval_set=[vw], verbose=False)
            _, tr_auc  = _ks_auc(y_train, clf.predict_proba(Xtr)[:, 1], tw)
            _, ts_auc  = _ks_auc(y_test,  clf.predict_proba(Xts)[:, 1], vw)
            _, oot_auc = _ks_auc(y_oot,   clf.predict_proba(Xoot)[:, 1], ow)
            loss = -oot_auc + abs(tr_auc - ts_auc)
            logs.append({"train_auc": tr_auc, "test_auc": ts_auc, "oot_auc": oot_auc, "loss": round(loss, 4)})
            return {"loss": loss, "status": STATUS_OK}

        space  = custom_space if custom_space is not None else self.get_default_space(odds)
        trials = Trials()
        fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=self.n_trials, trials=trials, verbose=False)
        self.trials_log = pd.DataFrame(logs)

        best_idx = self.trials_log["loss"].idxmin()
        best_t   = trials.trials[best_idx]
        p1 = {k: v[0] if isinstance(v, list) else v for k, v in best_t["misc"]["vals"].items()}
        p1.update({"base_score": 0.5, "booster": "gbtree", "objective": "binary:logistic",
                   "colsample_bytree": 1, "n_estimators": 500, "random_state": self.random_state,
                   "tree_method": tree_method, "early_stopping_rounds": 30, "eval_metric": "auc", "nthread": -1})
        p1["max_depth"] = [2, 3][int(p1.get("max_depth", 0))]
        model1 = xgb.XGBClassifier(**p1, verbosity=0)
        model1.fit(Xtr, y_train, sample_weight=tw, eval_set=[(Xts, y_test)],
                   sample_weight_eval_set=[vw], verbose=False)

        imp = pd.Series(model1.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
        self.selected_features = imp.index.tolist() if len(imp) > 0 else features

        Xtr2, Xts2, Xoot2 = Xtr[self.selected_features], Xts[self.selected_features], Xoot[self.selected_features]
        logs2 = []

        def objective2(params):
            clf = xgb.XGBClassifier(**params, verbosity=0)
            clf.fit(Xtr2, y_train, sample_weight=tw,
                    eval_set=[(Xts2, y_test)], sample_weight_eval_set=[vw], verbose=False)
            _, tr_auc  = _ks_auc(y_train, clf.predict_proba(Xtr2)[:, 1], tw)
            _, oot_auc = _ks_auc(y_oot,   clf.predict_proba(Xoot2)[:, 1], ow)
            loss = -oot_auc + abs(tr_auc - oot_auc)
            logs2.append(loss)
            return {"loss": loss, "status": STATUS_OK}

        space2 = dict(space)
        space2["n_estimators"]     = 1000
        space2["min_child_weight"] = hp.quniform("min_child_weight", 20, 1000, 10)
        trials2 = Trials()
        fmin(fn=objective2, space=space2, algo=tpe.suggest, max_evals=self.n_trials, trials=trials2, verbose=False)

        best_idx2 = int(np.argmin(logs2))
        best_t2   = trials2.trials[best_idx2]
        p2 = {k: v[0] if isinstance(v, list) else v for k, v in best_t2["misc"]["vals"].items()}
        p2.update({"base_score": 0.5, "booster": "gbtree", "objective": "binary:logistic",
                   "colsample_bytree": 1, "n_estimators": 1000, "random_state": self.random_state,
                   "tree_method": tree_method, "early_stopping_rounds": 30, "eval_metric": "auc", "nthread": -1})
        p2["max_depth"] = [2, 3][int(p2.get("max_depth", 0))]
        self.model = xgb.XGBClassifier(**p2, verbosity=0)
        self.model.fit(Xtr2, y_train, sample_weight=tw, eval_set=[(Xts2, y_test)],
                       sample_weight_eval_set=[vw], verbose=False)
        self.best_params = p2

        if save_dir:
            self.save(save_dir)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        feats = self.selected_features or X.columns.tolist()
        return self.model.predict_proba(X[feats])[:, 1]

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_model(os.path.join(save_dir, "xgb_model.json"))
        with open(os.path.join(save_dir, "selected_features.json"), "w", encoding="utf-8") as f:
            json.dump(self.selected_features, f, ensure_ascii=False, indent=2)
        meta = {"model_type": "xgboost", "best_params": self.best_params,
                "selected_features": self.selected_features}
        with open(os.path.join(save_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"模型已保存至: {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> "XGBTrainer":
        import xgboost as xgb
        trainer = cls()
        trainer.model = xgb.XGBClassifier()
        trainer.model.load_model(os.path.join(save_dir, "xgb_model.json"))
        with open(os.path.join(save_dir, "selected_features.json"), encoding="utf-8") as f:
            trainer.selected_features = json.load(f)
        with open(os.path.join(save_dir, "model_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        trainer.best_params = meta.get("best_params", {})
        return trainer


def _rmse(y_true, y_pred, weights=None):
    if weights is not None:
        weights = np.array(weights)
        weights = weights / weights.sum()
        return round(float(np.sqrt(np.sum(weights * (np.array(y_true) - np.array(y_pred)) ** 2))), 4)
    return round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)


def _reg_loss_mae(y_tr, p_tr, y_ts, p_ts, y_oo, p_oo, tw_, vw_, ow_):
    """
    共用回归调参 loss（LGBM 回归 / XGBoost 回归均使用同一函数）：
      loss = oot_mae
           + 0.5 * (oot_mae / tr_mae)           # OOT 相对训练集的衰退倍数（无量纲）
           + 0.3 * |ts_mae - oot_mae| / ts_mae  # test→OOT 时间漂移率（无量纲）
    主指标 MAE 对异常值鲁棒；后两项无量纲化，防止量纲差异导致惩罚项失效。
    """
    from sklearn.metrics import mean_absolute_error
    tr_mae  = mean_absolute_error(y_tr, p_tr,  sample_weight=tw_)
    ts_mae  = mean_absolute_error(y_ts, p_ts,  sample_weight=vw_)
    oot_mae = mean_absolute_error(y_oo, p_oo,  sample_weight=ow_)
    oot_rel = oot_mae / (tr_mae + 1e-8)
    drift   = abs(ts_mae - oot_mae) / (ts_mae + 1e-8)
    loss    = oot_mae + 0.5 * oot_rel + 0.3 * drift
    return round(loss, 6), round(tr_mae, 4), round(ts_mae, 4), round(oot_mae, 4)


class LGBMRegressorTrainer:
    """LightGBM 回归训练器，两轮 Hyperopt 调参 + gain 重要性特征筛选
    调参 loss 与 XGBRegressorTrainer 完全一致：
      oot_mae + 0.5*(oot_mae/tr_mae) + 0.3*|ts_mae-oot_mae|/ts_mae
    """

    def __init__(
        self,
        n_trials: int = 100,
        cat_cols: Optional[list] = None,
        random_state: int = 2024,
    ):
        self.n_trials = n_trials
        self.cat_cols = cat_cols or []
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.selected_features: list = []
        self.trials_log: pd.DataFrame = pd.DataFrame()

    def get_default_space(self, n_data: int) -> dict:
        """
        返回默认超参数搜索空间字典。

        在 Jupyter 中不重启修改参数的用法：
            space = trainer.get_default_space(len(X_train))
            from hyperopt import hp
            space['learning_rate'] = hp.uniform('learning_rate', 0.001, 0.05)
            trainer.fit(..., custom_space=space)
        """
        from hyperopt import hp
        return {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "mae",
            "feature_pre_filter": True,
            "nthread": -1,
            "verbose": -1,
            "bagging_freq": 2,
            "bagging_fraction": hp.quniform("bagging_fraction", 0.5, 0.8, 0.1),
            "feature_fraction": hp.quniform("feature_fraction", 0.5, 0.8, 0.1),
            "max_depth": hp.quniform("max_depth", 3, 6, 1),
            "num_leaves": hp.quniform("num_leaves", 8, 64, 4),
            "n_estimators": hp.quniform("n_estimators", 300, 600, 10),
            "learning_rate": hp.uniform("learning_rate", 0.005, 0.05),
            "lambda_l1": hp.randint("lambda_l1", 0, 50),
            "lambda_l2": hp.randint("lambda_l2", 0, 50),
            "min_data_in_leaf": hp.quniform(
                "min_data_in_leaf",
                max(int(n_data * 0.01), 1),
                max(int(n_data * 0.05), 2),
                50,
            ),
        }

    def fit(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test:  pd.DataFrame, y_test:  pd.Series,
        X_oot:   pd.DataFrame, y_oot:   pd.Series,
        train_weight: Optional[pd.Series] = None,
        test_weight:  Optional[pd.Series] = None,
        oot_weight:   Optional[pd.Series] = None,
        features: Optional[list] = None,
        custom_space: Optional[dict] = None,
        save_dir: Optional[str] = None,
    ) -> "LGBMRegressorTrainer":
        try:
            import lightgbm as lgb
            from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
        except ImportError as e:
            raise ImportError(f"请安装依赖: pip install lightgbm hyperopt — {e}")

        features = features or X_train.columns.tolist()
        Xtr  = X_train[features].reset_index(drop=True)
        Xts  = X_test[features].reset_index(drop=True)
        Xoot = X_oot[features].reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test  = y_test.reset_index(drop=True)
        y_oot   = y_oot.reset_index(drop=True)
        tw = train_weight.reset_index(drop=True) if train_weight is not None else pd.Series(np.ones(len(Xtr)))
        vw = test_weight.reset_index(drop=True)  if test_weight  is not None else pd.Series(np.ones(len(Xts)))
        ow = oot_weight.reset_index(drop=True)   if oot_weight   is not None else pd.Series(np.ones(len(Xoot)))

        train_set = lgb.Dataset(Xtr, y_train, categorical_feature=self.cat_cols, weight=tw, free_raw_data=False)
        test_set  = lgb.Dataset(Xts, y_test,  categorical_feature=self.cat_cols, weight=vw, free_raw_data=False)

        logs = []

        def objective(params):
            params["num_leaves"]       = int(params["num_leaves"])
            params["max_depth"]        = int(params["max_depth"])
            num_boost_round            = int(params.pop("n_estimators"))
            params["lambda_l1"]        = int(params["lambda_l1"])
            params["lambda_l2"]        = int(params["lambda_l2"])
            params["min_data_in_leaf"] = int(params["min_data_in_leaf"])
            m = lgb.train(params, train_set, num_boost_round=num_boost_round, valid_sets=test_set,
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            loss, tr_mae, ts_mae, oot_mae = _reg_loss_mae(
                y_train.values, m.predict(Xtr, num_iteration=m.best_iteration),
                y_test.values,  m.predict(Xts, num_iteration=m.best_iteration),
                y_oot.values,   m.predict(Xoot, num_iteration=m.best_iteration),
                tw.values, vw.values, ow.values,
            )
            logs.append({"train_mae": tr_mae, "test_mae": ts_mae, "oot_mae": oot_mae,
                         "loss": loss, "params": {**params, "n_estimators": num_boost_round}})
            return {"loss": loss, "status": STATUS_OK}

        n_data = len(Xtr)
        space  = custom_space if custom_space is not None else self.get_default_space(n_data)
        trials = Trials()
        fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=self.n_trials, trials=trials, verbose=False)
        self.trials_log = pd.DataFrame(logs)

        # ── 第一轮最优参数重新训练，用于特征筛选 ──────────────────────────────
        best_row = self.trials_log.loc[self.trials_log["loss"].idxmin(), "params"]
        params1  = dict(best_row)
        num_br1  = int(params1.pop("n_estimators", 300))
        model1   = lgb.train(params1, train_set, num_boost_round=num_br1, valid_sets=test_set,
                             callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

        imp = pd.DataFrame({"feature": model1.feature_name(), "importance": model1.feature_importance("gain")})
        self.selected_features = imp[imp["importance"] > 0]["feature"].tolist()
        if not self.selected_features:
            self.selected_features = features

        # ── 第二轮：筛选后特征，精细调参 ────────────────────────────────────
        Xtr2, Xts2, Xoot2 = Xtr[self.selected_features], Xts[self.selected_features], Xoot[self.selected_features]
        train_set2 = lgb.Dataset(Xtr2, y_train, categorical_feature=self.cat_cols, weight=tw, free_raw_data=False)
        test_set2  = lgb.Dataset(Xts2, y_test,  categorical_feature=self.cat_cols, weight=vw, free_raw_data=False)

        logs2 = []

        def objective2(params):
            params["num_leaves"]       = int(params["num_leaves"])
            params["max_depth"]        = int(params["max_depth"])
            num_boost_round2           = int(params.pop("n_estimators"))
            params["lambda_l1"]        = int(params["lambda_l1"])
            params["lambda_l2"]        = int(params["lambda_l2"])
            params["min_data_in_leaf"] = int(params["min_data_in_leaf"])
            m = lgb.train(params, train_set2, num_boost_round=num_boost_round2, valid_sets=test_set2,
                          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
            loss, tr_mae, ts_mae, oot_mae = _reg_loss_mae(
                y_train.values, m.predict(Xtr2, num_iteration=m.best_iteration),
                y_test.values,  m.predict(Xts2, num_iteration=m.best_iteration),
                y_oot.values,   m.predict(Xoot2, num_iteration=m.best_iteration),
                tw.values, vw.values, ow.values,
            )
            logs2.append({"train_mae": tr_mae, "test_mae": ts_mae, "oot_mae": oot_mae, "loss": loss})
            return {"loss": loss, "status": STATUS_OK}

        space2 = dict(space)
        space2["feature_pre_filter"] = False
        space2["bagging_fraction"]   = hp.quniform("bagging_fraction", 0.6, 0.8, 0.1)
        space2["feature_fraction"]   = hp.quniform("feature_fraction", 0.6, 0.8, 0.1)
        space2["n_estimators"]       = hp.quniform("n_estimators", 500, 1000, 10)
        space2["min_data_in_leaf"]   = hp.quniform("min_data_in_leaf", 1, max(int(n_data * 0.03), 2), 1)
        trials2 = Trials()
        fmin(fn=objective2, space=space2, algo=tpe.suggest, max_evals=self.n_trials, trials=trials2, verbose=False)
        self.trials_log = pd.DataFrame(logs + logs2)

        idx2    = int(np.argmin([r["loss"] for r in logs2]))
        params2 = trials2.trials[idx2]["misc"]["vals"]
        params2 = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in params2.items()}
        num_br2 = int(params2.pop("n_estimators", 500))
        params2.update({
            "boosting_type": "gbdt", "objective": "regression", "metric": "mae",
            "feature_pre_filter": False, "nthread": -1, "verbose": -1, "bagging_freq": 2,
        })
        for k in ["num_leaves", "max_depth", "lambda_l1", "lambda_l2", "min_data_in_leaf"]:
            if k in params2:
                params2[k] = int(params2[k])
        self.model = lgb.train(params2, train_set2, num_boost_round=num_br2, valid_sets=test_set2,
                               callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        self.best_params = {**params2, "n_estimators": num_br2}

        if save_dir:
            self.save(save_dir)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        feats = self.selected_features or X.columns.tolist()
        return self.model.predict(X[feats], num_iteration=self.model.best_iteration)

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_model(os.path.join(save_dir, "lgbm_reg_model.txt"))
        with open(os.path.join(save_dir, "selected_features.json"), "w", encoding="utf-8") as f:
            json.dump(self.selected_features, f, ensure_ascii=False, indent=2)
        meta = {"model_type": "lgbm_reg", "best_params": self.best_params,
                "selected_features": self.selected_features}
        with open(os.path.join(save_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"模型已保存至: {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> "LGBMRegressorTrainer":
        import lightgbm as lgb
        trainer = cls()
        trainer.model = lgb.Booster(model_file=os.path.join(save_dir, "lgbm_reg_model.txt"))
        with open(os.path.join(save_dir, "selected_features.json"), encoding="utf-8") as f:
            trainer.selected_features = json.load(f)
        with open(os.path.join(save_dir, "model_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        trainer.best_params = meta.get("best_params", {})
        return trainer


class XGBRegressorTrainer:
    """XGBoost 回归训练器"""

    def __init__(self, n_trials: int = 100, use_gpu: bool = False, random_state: int = 2024):
        self.n_trials = n_trials
        self.use_gpu = use_gpu
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.selected_features: list = []
        self.trials_log: pd.DataFrame = pd.DataFrame()

    def get_default_space(self) -> dict:
        """
        返回默认超参数搜索空间字典。

        在 Jupyter 中不重启修改参数的用法：
            space = trainer.get_default_space()
            from hyperopt import hp
            space['max_depth'] = hp.choice('max_depth', [3, 4, 5, 6, 7])
            trainer.fit(..., custom_space=space)
        """
        from hyperopt import hp
        return {
            "objective": "reg:squarederror", "booster": "gbtree", "eval_metric": "rmse",
            "learning_rate": hp.quniform("learning_rate", 0.01, 0.3, 0.01),
            "gamma": hp.quniform("gamma", 0, 50, 2),
            "max_depth": hp.choice("max_depth", [3, 4, 5, 6]),
            "subsample": hp.quniform("subsample", 0.5, 1.0, 0.1),
            "colsample_bytree": hp.quniform("colsample_bytree", 0.5, 1.0, 0.1),
            "n_estimators": 500,
            "min_child_weight": hp.quniform("min_child_weight", 5, 100, 5),
            "reg_alpha": hp.quniform("reg_alpha", 0, 100, 5),
            "reg_lambda": hp.quniform("reg_lambda", 0, 100, 5),
            "random_state": self.random_state,
            "tree_method": "gpu_hist" if self.use_gpu else "hist",
            "early_stopping_rounds": 30,
            "nthread": -1,
        }

    def fit(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test:  pd.DataFrame, y_test:  pd.Series,
        X_oot:   pd.DataFrame, y_oot:   pd.Series,
        train_weight: Optional[pd.Series] = None,
        test_weight:  Optional[pd.Series] = None,
        oot_weight:   Optional[pd.Series] = None,
        features: Optional[list] = None,
        custom_space: Optional[dict] = None,
        save_dir: Optional[str] = None,
    ) -> "XGBRegressorTrainer":
        try:
            import xgboost as xgb
            from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
        except ImportError as e:
            raise ImportError(f"请安装依赖: pip install xgboost hyperopt — {e}")

        features = features or X_train.columns.tolist()
        Xtr  = X_train[features].reset_index(drop=True)
        Xts  = X_test[features].reset_index(drop=True)
        Xoot = X_oot[features].reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test  = y_test.reset_index(drop=True)
        y_oot   = y_oot.reset_index(drop=True)
        tw = train_weight.reset_index(drop=True) if train_weight is not None else pd.Series(np.ones(len(Xtr)))
        vw = test_weight.reset_index(drop=True)  if test_weight  is not None else pd.Series(np.ones(len(Xts)))
        ow = oot_weight.reset_index(drop=True)   if oot_weight   is not None else pd.Series(np.ones(len(Xoot)))

        tree_method = "gpu_hist" if self.use_gpu else "hist"
        logs = []

        def objective(params):
            reg = xgb.XGBRegressor(**params, verbosity=0)
            reg.fit(Xtr, y_train, sample_weight=tw,
                    eval_set=[(Xts, y_test)], sample_weight_eval_set=[vw], verbose=False)
            loss, tr_mae, ts_mae, oot_mae = _reg_loss_mae(
                y_train, reg.predict(Xtr), y_test, reg.predict(Xts), y_oot, reg.predict(Xoot),
                tw.values, vw.values, ow.values,
            )
            logs.append({"train_mae": tr_mae, "test_mae": ts_mae, "oot_mae": oot_mae, "loss": loss})
            return {"loss": loss, "status": STATUS_OK}

        space  = custom_space if custom_space is not None else self.get_default_space()
        trials = Trials()
        fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=self.n_trials,
             trials=trials, verbose=False)
        self.trials_log = pd.DataFrame(logs)   # 第一轮暂存，第二轮后合并

        best_idx = self.trials_log["loss"].idxmin()
        best_t   = trials.trials[best_idx]
        p1 = {k: v[0] if isinstance(v, list) else v for k, v in best_t["misc"]["vals"].items()}
        p1.update({"objective": "reg:squarederror", "booster": "gbtree", "eval_metric": "mae",
                   "colsample_bytree": p1.get("colsample_bytree", 1.0), "n_estimators": 500,
                   "random_state": self.random_state, "tree_method": tree_method,
                   "early_stopping_rounds": 30, "nthread": -1})
        p1["max_depth"] = [3, 4, 5, 6][int(p1.get("max_depth", 0))]
        model1 = xgb.XGBRegressor(**p1, verbosity=0)
        model1.fit(Xtr, y_train, sample_weight=tw, eval_set=[(Xts, y_test)],
                   sample_weight_eval_set=[vw], verbose=False)

        imp = pd.Series(model1.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
        self.selected_features = imp.index.tolist() if len(imp) > 0 else features

        Xtr2, Xts2, Xoot2 = Xtr[self.selected_features], Xts[self.selected_features], Xoot[self.selected_features]
        logs2 = []

        def objective2(params):
            reg = xgb.XGBRegressor(**params, verbosity=0)
            reg.fit(Xtr2, y_train, sample_weight=tw,
                    eval_set=[(Xts2, y_test)], sample_weight_eval_set=[vw], verbose=False)
            loss, tr_mae, ts_mae, oot_mae = _reg_loss_mae(
                y_train, reg.predict(Xtr2), y_test, reg.predict(Xts2), y_oot, reg.predict(Xoot2),
                tw.values, vw.values, ow.values,
            )
            logs2.append({"train_mae": tr_mae, "test_mae": ts_mae, "oot_mae": oot_mae, "loss": loss})
            return {"loss": loss, "status": STATUS_OK}

        space2 = dict(space)
        space2["n_estimators"]     = 1000
        space2["min_child_weight"] = hp.quniform("min_child_weight", 1, 100, 1)
        trials2 = Trials()
        fmin(fn=objective2, space=space2, algo=tpe.suggest, max_evals=self.n_trials,
             trials=trials2, verbose=False)
        self.trials_log = pd.DataFrame(logs + logs2)

        best_idx2 = int(np.argmin([r["loss"] for r in logs2]))
        best_t2   = trials2.trials[best_idx2]
        p2 = {k: v[0] if isinstance(v, list) else v for k, v in best_t2["misc"]["vals"].items()}
        p2.update({"objective": "reg:squarederror", "booster": "gbtree", "eval_metric": "mae",
                   "n_estimators": 1000, "random_state": self.random_state,
                   "tree_method": tree_method, "early_stopping_rounds": 30, "nthread": -1})
        p2["max_depth"] = [3, 4, 5, 6][int(p2.get("max_depth", 0))]
        self.model = xgb.XGBRegressor(**p2, verbosity=0)
        self.model.fit(Xtr2, y_train, sample_weight=tw, eval_set=[(Xts2, y_test)],
                       sample_weight_eval_set=[vw], verbose=False)
        self.best_params = p2

        if save_dir:
            self.save(save_dir)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        feats = self.selected_features or X.columns.tolist()
        return self.model.predict(X[feats])

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_model(os.path.join(save_dir, "xgb_reg_model.json"))
        with open(os.path.join(save_dir, "selected_features.json"), "w", encoding="utf-8") as f:
            json.dump(self.selected_features, f, ensure_ascii=False, indent=2)
        meta = {"model_type": "xgboost_reg", "best_params": self.best_params,
                "selected_features": self.selected_features}
        with open(os.path.join(save_dir, "model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"模型已保存至: {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> "XGBRegressorTrainer":
        import xgboost as xgb
        trainer = cls()
        trainer.model = xgb.XGBRegressor()
        trainer.model.load_model(os.path.join(save_dir, "xgb_reg_model.json"))
        with open(os.path.join(save_dir, "selected_features.json"), encoding="utf-8") as f:
            trainer.selected_features = json.load(f)
        with open(os.path.join(save_dir, "model_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        trainer.best_params = meta.get("best_params", {})
        return trainer


class ModelTrainer:
    """统一训练入口，支持 lgbm / xgboost / xgboost_reg / lgbm_reg"""

    def __init__(self, model_type: str = "lgbm", **kwargs):
        self.model_type = model_type
        if model_type == "lgbm":
            self._trainer = LGBMTrainer(**kwargs)
        elif model_type == "xgboost":
            self._trainer = XGBTrainer(**kwargs)
        elif model_type == "xgboost_reg":
            self._trainer = XGBRegressorTrainer(**kwargs)
        elif model_type == "lgbm_reg":
            self._trainer = LGBMRegressorTrainer(**kwargs)
        else:
            raise ValueError(f"model_type 须为 'lgbm' / 'xgboost' / 'xgboost_reg' / 'lgbm_reg'，不支持 '{model_type}'")

    def get_default_space(self, **kwargs) -> dict:
        """获取当前模型类型的默认参数空间，可在 notebook 中修改后传给 fit(custom_space=...)"""
        return self._trainer.get_default_space(**kwargs)

    def fit(self, X_train, y_train, X_test, y_test, X_oot, y_oot,
            custom_space=None, save_dir=None, **kwargs) -> "ModelTrainer":
        self._trainer.fit(X_train, y_train, X_test, y_test, X_oot, y_oot,
                          custom_space=custom_space, save_dir=save_dir, **kwargs)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_type in ("xgboost_reg", "lgbm_reg"):
            raise TypeError(f"{self.model_type} 是回归模型，请使用 predict()")
        return self._trainer.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_type not in ("xgboost_reg", "lgbm_reg"):
            raise TypeError(f"{self.model_type} 是分类模型，请使用 predict_proba()")
        return self._trainer.predict(X)

    def save(self, save_dir: str) -> None:
        """保存模型文件、入模特征和参数元信息到 save_dir"""
        self._trainer.save(save_dir)

    @classmethod
    def load(cls, save_dir: str) -> "ModelTrainer":
        """从 save_dir 恢复模型，自动识别模型类型"""
        meta_path = os.path.join(save_dir, "model_meta.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        model_type = meta["model_type"]
        instance = cls.__new__(cls)
        instance.model_type = model_type
        if model_type == "lgbm":
            instance._trainer = LGBMTrainer.load(save_dir)
        elif model_type == "xgboost":
            instance._trainer = XGBTrainer.load(save_dir)
        elif model_type == "xgboost_reg":
            instance._trainer = XGBRegressorTrainer.load(save_dir)
        elif model_type == "lgbm_reg":
            instance._trainer = LGBMRegressorTrainer.load(save_dir)
        else:
            raise ValueError(f"未知 model_type: {model_type}")
        return instance

    @property
    def selected_features(self):
        return self._trainer.selected_features

    @property
    def best_params(self):
        return self._trainer.best_params

    @property
    def trials_log(self):
        return self._trainer.trials_log
