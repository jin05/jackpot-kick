"""Phase 3: Model Training Module.

LightGBMを使用したサッカー試合結果予測モデル。
"""

import pickle
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold


class TotoModel:
    """LightGBMを使用したサッカー試合結果予測モデル.

    Optunaによるハイパーパラメータ最適化と
    StratifiedKFoldによるクロスバリデーションを行う。
    """

    # 特徴量として使用するカラム
    FEATURE_COLUMNS = [
        "home_elo",
        "away_elo",
        "home_recent_goals_avg",
        "home_recent_conceded_avg",
        "home_recent_win_rate",
        "away_recent_goals_avg",
        "away_recent_conceded_avg",
        "away_recent_win_rate",
        "home_team_encoded",
        "away_team_encoded",
    ]

    TARGET_COLUMN = "result"

    def __init__(
        self,
        n_trials: int = 50,
        n_splits: int = 5,
        random_state: int = 42,
        verbose: bool = True,
    ):
        """TotoModelの初期化.

        Args:
            n_trials: Optunaの試行回数
            n_splits: クロスバリデーションの分割数
            random_state: 乱数シード
            verbose: 詳細出力の有無
        """
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.random_state = random_state
        self.verbose = verbose
        self.model: Optional[lgb.LGBMClassifier] = None
        self.best_params: Optional[dict] = None
        self.cv_results: Optional[dict] = None

    def _create_objective(
        self, X: np.ndarray, y: np.ndarray
    ) -> callable:
        """Optuna用の目的関数を作成.

        Args:
            X: 特徴量配列
            y: ターゲット配列

        Returns:
            目的関数
        """

        def objective(trial: optuna.Trial) -> float:
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "verbosity": -1,
                "random_state": self.random_state,
                "n_jobs": -1,
                # ハイパーパラメータ探索範囲
                "num_leaves": trial.suggest_int("num_leaves", 10, 100),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            # StratifiedKFoldでクロスバリデーション
            skf = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state,
            )

            log_losses = []
            for train_idx, val_idx in skf.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=20, verbose=False),
                    ],
                )

                y_pred_proba = model.predict_proba(X_val)
                fold_log_loss = log_loss(y_val, y_pred_proba)
                log_losses.append(fold_log_loss)

            return np.mean(log_losses)

        return objective

    def optimize_hyperparameters(
        self, X: np.ndarray, y: np.ndarray
    ) -> dict:
        """Optunaでハイパーパラメータを最適化.

        Args:
            X: 特徴量配列
            y: ターゲット配列

        Returns:
            最適なハイパーパラメータ
        """
        if self.verbose:
            print("\nOptimizing hyperparameters with Optuna...")
            print(f"  Trials: {self.n_trials}")
            print(f"  CV Folds: {self.n_splits}")

        # Optunaのログレベルを設定
        optuna.logging.set_verbosity(
            optuna.logging.INFO if self.verbose else optuna.logging.WARNING
        )

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        objective = self._create_objective(X, y)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose)

        self.best_params = study.best_params

        if self.verbose:
            print(f"\nBest trial:")
            print(f"  Value (log_loss): {study.best_value:.4f}")
            print(f"  Best params:")
            for key, value in self.best_params.items():
                print(f"    {key}: {value}")

        return self.best_params

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, params: Optional[dict] = None
    ) -> dict:
        """StratifiedKFoldでクロスバリデーションを実行.

        Args:
            X: 特徴量配列
            y: ターゲット配列
            params: モデルパラメータ（Noneの場合はbest_paramsを使用）

        Returns:
            クロスバリデーション結果（accuracy, log_loss）
        """
        if params is None:
            if self.best_params is None:
                raise ValueError(
                    "パラメータが指定されていません。"
                    "optimize_hyperparameters()を先に実行するか、paramsを指定してください。"
                )
            params = self.best_params

        # 基本パラメータを追加
        full_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": self.random_state,
            "n_jobs": -1,
            **params,
        }

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        accuracies = []
        log_losses = []

        if self.verbose:
            print(f"\nCross-validation with {self.n_splits} folds...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**full_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20, verbose=False),
                ],
            )

            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)

            fold_accuracy = accuracy_score(y_val, y_pred)
            fold_log_loss = log_loss(y_val, y_pred_proba)

            accuracies.append(fold_accuracy)
            log_losses.append(fold_log_loss)

            if self.verbose:
                print(
                    f"  Fold {fold}: accuracy={fold_accuracy:.4f}, "
                    f"log_loss={fold_log_loss:.4f}"
                )

        self.cv_results = {
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "log_loss_mean": np.mean(log_losses),
            "log_loss_std": np.std(log_losses),
            "fold_accuracies": accuracies,
            "fold_log_losses": log_losses,
        }

        if self.verbose:
            print(f"\nCross-validation results:")
            print(
                f"  Accuracy: {self.cv_results['accuracy_mean']:.4f} "
                f"(+/- {self.cv_results['accuracy_std']:.4f})"
            )
            print(
                f"  Log Loss: {self.cv_results['log_loss_mean']:.4f} "
                f"(+/- {self.cv_results['log_loss_std']:.4f})"
            )

        return self.cv_results

    def fit(
        self, X: np.ndarray, y: np.ndarray, optimize: bool = True
    ) -> "TotoModel":
        """モデルを学習.

        Args:
            X: 特徴量配列
            y: ターゲット配列
            optimize: ハイパーパラメータ最適化を行うかどうか

        Returns:
            学習済みモデル（self）
        """
        if optimize:
            self.optimize_hyperparameters(X, y)

        if self.best_params is None:
            # デフォルトパラメータを使用
            self.best_params = {
                "num_leaves": 31,
                "learning_rate": 0.1,
                "max_depth": 6,
                "n_estimators": 100,
            }

        # クロスバリデーションで評価
        self.cross_validate(X, y)

        # 最終モデルを全データで学習
        full_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": self.random_state,
            "n_jobs": -1,
            **self.best_params,
        }

        if self.verbose:
            print("\nTraining final model on full dataset...")

        self.model = lgb.LGBMClassifier(**full_params)
        self.model.fit(X, y)

        if self.verbose:
            print("Model training completed.")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """予測を実行.

        Args:
            X: 特徴量配列

        Returns:
            予測結果（クラスラベル）
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。fit()を先に実行してください。")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """クラス確率を予測.

        Args:
            X: 特徴量配列

        Returns:
            クラス確率
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。fit()を先に実行してください。")
        return self.model.predict_proba(X)

    def save(self, path: str | Path) -> None:
        """モデルを保存.

        Args:
            path: 保存先パス
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。fit()を先に実行してください。")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model,
            "best_params": self.best_params,
            "cv_results": self.cv_results,
            "feature_columns": self.FEATURE_COLUMNS,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        if self.verbose:
            print(f"\nModel saved to: {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TotoModel":
        """保存されたモデルを読み込み.

        Args:
            path: モデルファイルのパス

        Returns:
            読み込んだTotoModelインスタンス
        """
        path = Path(path)

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        instance = cls(
            n_splits=model_data.get("n_splits", 5),
            random_state=model_data.get("random_state", 42),
        )
        instance.model = model_data["model"]
        instance.best_params = model_data["best_params"]
        instance.cv_results = model_data.get("cv_results")

        return instance

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """DataFrameから特徴量を抽出.

        Args:
            df: 処理済みDataFrame

        Returns:
            特徴量配列
        """
        # NaN値を含む行を除外するためのマスク
        mask = df[self.FEATURE_COLUMNS].notna().all(axis=1)
        return df.loc[mask, self.FEATURE_COLUMNS].values

    def prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """DataFrameからターゲットを抽出.

        Args:
            df: 処理済みDataFrame

        Returns:
            ターゲット配列
        """
        mask = df[self.FEATURE_COLUMNS].notna().all(axis=1)
        return df.loc[mask, self.TARGET_COLUMN].values

    def get_feature_importance(self) -> pd.DataFrame:
        """特徴量の重要度を取得.

        Returns:
            特徴量重要度のDataFrame
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません。fit()を先に実行してください。")

        importance_df = pd.DataFrame(
            {
                "feature": self.FEATURE_COLUMNS,
                "importance": self.model.feature_importances_,
            }
        )
        return importance_df.sort_values("importance", ascending=False).reset_index(
            drop=True
        )
