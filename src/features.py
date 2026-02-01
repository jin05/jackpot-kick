"""
特徴量エンジニアリングモジュール

ELO Rating、直近フォーム、エンコーディング等の特徴量を計算する。
"""

from pathlib import Path
from collections import defaultdict

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """
    試合データに対して特徴量エンジニアリングを行うクラス

    以下の特徴量を追加:
    - ELO Rating: 各チームのレーティング（初期値1500）
    - Recent Form: 直近5試合の移動平均（得点、失点、勝率）
    - Label Encoding: チーム名の数値エンコーディング
    """

    def __init__(
        self,
        initial_elo: float = 1500.0,
        k_factor: float = 32.0,
        recent_n: int = 5,
    ):
        """
        Args:
            initial_elo: ELOレーティングの初期値
            k_factor: ELO計算のK係数
            recent_n: 直近フォーム計算に使う試合数
        """
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.recent_n = recent_n

        # チームごとのELOレーティングを管理
        self._elo_ratings: dict[str, float] = defaultdict(lambda: self.initial_elo)

        # チームごとの直近試合履歴を管理
        self._team_history: dict[str, list[dict]] = defaultdict(list)

        # Label Encoderのインスタンス
        self._label_encoder: LabelEncoder | None = None

    def _calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        ELO計算における期待勝率を計算

        Args:
            rating_a: チームAのレーティング
            rating_b: チームBのレーティング

        Returns:
            チームAの期待勝率（0.0〜1.0）
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _get_actual_score(self, result: int, is_home: bool) -> float:
        """
        試合結果から実際のスコアを取得

        Args:
            result: 試合結果（0=Away Win, 1=Draw, 2=Home Win）
            is_home: ホームチームかどうか

        Returns:
            実際のスコア（勝利=1.0, 引き分け=0.5, 敗北=0.0）
        """
        if result == 1:  # Draw
            return 0.5
        elif result == 2:  # Home Win
            return 1.0 if is_home else 0.0
        else:  # Away Win
            return 0.0 if is_home else 1.0

    def _update_elo(
        self, home_team: str, away_team: str, result: int
    ) -> tuple[float, float]:
        """
        試合結果に基づいてELOレーティングを更新

        Args:
            home_team: ホームチーム名
            away_team: アウェイチーム名
            result: 試合結果

        Returns:
            更新後の（ホームELO, アウェイELO）
        """
        home_elo = self._elo_ratings[home_team]
        away_elo = self._elo_ratings[away_team]

        # 期待スコアを計算
        expected_home = self._calculate_expected_score(home_elo, away_elo)
        expected_away = 1.0 - expected_home

        # 実際のスコアを取得
        actual_home = self._get_actual_score(result, is_home=True)
        actual_away = self._get_actual_score(result, is_home=False)

        # ELOを更新
        new_home_elo = home_elo + self.k_factor * (actual_home - expected_home)
        new_away_elo = away_elo + self.k_factor * (actual_away - expected_away)

        self._elo_ratings[home_team] = new_home_elo
        self._elo_ratings[away_team] = new_away_elo

        return new_home_elo, new_away_elo

    def _get_recent_form(self, team: str) -> dict[str, float | None]:
        """
        チームの直近N試合のフォームを計算

        Args:
            team: チーム名

        Returns:
            直近フォームの統計情報
        """
        history = self._team_history[team]

        if len(history) == 0:
            return {
                "goals_avg": None,
                "conceded_avg": None,
                "win_rate": None,
            }

        # 直近N試合のみ使用
        recent = history[-self.recent_n :]

        goals = [h["goals"] for h in recent]
        conceded = [h["conceded"] for h in recent]
        wins = [1 if h["win"] else 0 for h in recent]

        return {
            "goals_avg": sum(goals) / len(goals),
            "conceded_avg": sum(conceded) / len(conceded),
            "win_rate": sum(wins) / len(wins),
        }

    def _update_team_history(
        self,
        team: str,
        goals: int,
        conceded: int,
        result: int,
        is_home: bool,
    ) -> None:
        """
        チームの試合履歴を更新

        Args:
            team: チーム名
            goals: 得点
            conceded: 失点
            result: 試合結果
            is_home: ホームチームかどうか
        """
        # 勝利判定
        if result == 1:  # Draw
            win = False
        elif result == 2:  # Home Win
            win = is_home
        else:  # Away Win
            win = not is_home

        self._team_history[team].append(
            {
                "goals": goals,
                "conceded": conceded,
                "win": win,
            }
        )

    def add_elo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ELO Rating特徴量を追加

        試合前のELOレーティングを記録し、試合後に更新する。

        Args:
            df: 試合データのDataFrame（date順にソート済み）

        Returns:
            home_elo, away_elo カラムが追加されたDataFrame
        """
        # ELOレーティングをリセット
        self._elo_ratings = defaultdict(lambda: self.initial_elo)

        home_elos = []
        away_elos = []

        for _, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]
            result = row["result"]

            # 試合前のELOを記録
            home_elos.append(self._elo_ratings[home_team])
            away_elos.append(self._elo_ratings[away_team])

            # 試合結果でELOを更新
            self._update_elo(home_team, away_team, result)

        df = df.copy()
        df["home_elo"] = home_elos
        df["away_elo"] = away_elos

        return df

    def add_recent_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        直近5試合の移動平均特徴量を追加

        Args:
            df: 試合データのDataFrame（date順にソート済み）

        Returns:
            直近フォーム特徴量が追加されたDataFrame
        """
        # 履歴をリセット
        self._team_history = defaultdict(list)

        home_goals_avg = []
        home_conceded_avg = []
        home_win_rate = []
        away_goals_avg = []
        away_conceded_avg = []
        away_win_rate = []

        for _, row in df.iterrows():
            home_team = row["home_team"]
            away_team = row["away_team"]

            # 試合前の直近フォームを記録
            home_form = self._get_recent_form(home_team)
            away_form = self._get_recent_form(away_team)

            home_goals_avg.append(home_form["goals_avg"])
            home_conceded_avg.append(home_form["conceded_avg"])
            home_win_rate.append(home_form["win_rate"])
            away_goals_avg.append(away_form["goals_avg"])
            away_conceded_avg.append(away_form["conceded_avg"])
            away_win_rate.append(away_form["win_rate"])

            # 試合結果で履歴を更新
            self._update_team_history(
                home_team,
                goals=row["home_score"],
                conceded=row["away_score"],
                result=row["result"],
                is_home=True,
            )
            self._update_team_history(
                away_team,
                goals=row["away_score"],
                conceded=row["home_score"],
                result=row["result"],
                is_home=False,
            )

        df = df.copy()
        df["home_recent_goals_avg"] = home_goals_avg
        df["home_recent_conceded_avg"] = home_conceded_avg
        df["home_recent_win_rate"] = home_win_rate
        df["away_recent_goals_avg"] = away_goals_avg
        df["away_recent_conceded_avg"] = away_conceded_avg
        df["away_recent_win_rate"] = away_win_rate

        return df

    def add_label_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        チーム名のLabel Encodingを追加

        Args:
            df: 試合データのDataFrame

        Returns:
            home_team_encoded, away_team_encoded カラムが追加されたDataFrame
        """
        df = df.copy()

        # 全チーム名を収集してエンコーダーをfit
        all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(all_teams)

        # エンコードを適用
        df["home_team_encoded"] = self._label_encoder.transform(df["home_team"])
        df["away_team_encoded"] = self._label_encoder.transform(df["away_team"])

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        全ての特徴量エンジニアリングを適用

        Args:
            df: 試合データのDataFrame

        Returns:
            全ての特徴量が追加されたDataFrame
        """
        # 日付順でソート
        df = df.sort_values("date").reset_index(drop=True)

        # 各特徴量を追加
        df = self.add_elo_features(df)
        df = self.add_recent_form_features(df)
        df = self.add_label_encoding(df)

        return df

    def get_team_labels(self) -> dict[int, str]:
        """
        エンコードされたラベルとチーム名のマッピングを取得

        Returns:
            {encoded_value: team_name} の辞書
        """
        if self._label_encoder is None:
            return {}
        return {
            i: label for i, label in enumerate(self._label_encoder.classes_)
        }


def load_raw_data(filepath: str | Path = "data/raw_match_data.csv") -> pd.DataFrame:
    """
    生データを読み込む

    Args:
        filepath: CSVファイルのパス

    Returns:
        試合データのDataFrame
    """
    return pd.read_csv(filepath)


def save_processed_data(
    df: pd.DataFrame,
    filepath: str | Path = "data/processed_features.csv",
) -> None:
    """
    処理済みデータを保存

    Args:
        df: 処理済みのDataFrame
        filepath: 保存先のパス
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved processed features to: {filepath}")
