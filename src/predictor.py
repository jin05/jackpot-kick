"""Phase 4: Prediction & Strategy Module.

学習済みモデルを使って次回のToto対象試合の予想を行う機能を提供する。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FeatureEngineer, load_raw_data
from .model import TotoModel
from .scraper import TEAM_NAME_MAP


@dataclass
class MatchOdds:
    """ブックメーカーオッズを表すデータクラス.

    Attributes:
        home_win: ホーム勝利のオッズ
        draw: 引き分けのオッズ
        away_win: アウェイ勝利のオッズ
    """

    home_win: float
    draw: float
    away_win: float


@dataclass
class MatchCard:
    """対戦カードを表すデータクラス.

    Attributes:
        home_team: ホームチーム名
        away_team: アウェイチーム名
        odds: ブックメーカーオッズ
    """

    home_team: str
    away_team: str
    odds: MatchOdds


@dataclass
class PredictionResult:
    """予測結果を表すデータクラス.

    Attributes:
        match: 対戦カード
        prob_home_win: ホーム勝利確率
        prob_draw: 引き分け確率
        prob_away_win: アウェイ勝利確率
        ev_home_win: ホーム勝利の期待値
        ev_draw: 引き分けの期待値
        ev_away_win: アウェイ勝利の期待値
    """

    match: MatchCard
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    ev_home_win: float
    ev_draw: float
    ev_away_win: float

    @property
    def best_value_bet(self) -> tuple[str, float] | None:
        """最も期待値の高いValue Betを返す.

        Returns:
            (ベット種別, 期待値) のタプル。EV > 0のものがない場合はNone
        """
        evs = [
            ("Home Win", self.ev_home_win),
            ("Draw", self.ev_draw),
            ("Away Win", self.ev_away_win),
        ]
        positive_evs = [(name, ev) for name, ev in evs if ev > 0]
        if not positive_evs:
            return None
        return max(positive_evs, key=lambda x: x[1])

    @property
    def value_bets(self) -> list[tuple[str, float]]:
        """全てのValue Bet（EV > 0）を返す.

        Returns:
            (ベット種別, 期待値) のリスト
        """
        evs = [
            ("Home Win", self.ev_home_win),
            ("Draw", self.ev_draw),
            ("Away Win", self.ev_away_win),
        ]
        return [(name, ev) for name, ev in evs if ev > 0]


class MatchPredictor:
    """試合予測と戦略立案を行うクラス.

    学習済みモデルと過去のデータを使用して、
    新しい対戦カードの結果を予測し、期待値に基づく戦略を提案する。
    """

    def __init__(
        self,
        model: TotoModel,
        engineer: FeatureEngineer,
        verbose: bool = True,
    ):
        """MatchPredictorの初期化.

        Args:
            model: 学習済みのTotoModel
            engineer: 過去データで初期化済みのFeatureEngineer
            verbose: 詳細出力の有無
        """
        self.model = model
        self.engineer = engineer
        self.verbose = verbose

    @classmethod
    def from_files(
        cls,
        model_path: str | Path = "models/lgbm_model.pkl",
        data_path: str | Path = "data/raw_match_data.csv",
        verbose: bool = True,
    ) -> "MatchPredictor":
        """ファイルからMatchPredictorを作成.

        Args:
            model_path: 学習済みモデルのパス
            data_path: 過去データのCSVパス
            verbose: 詳細出力の有無

        Returns:
            初期化済みのMatchPredictor
        """
        # モデルを読み込み
        model = TotoModel.load(model_path)

        # 過去データを読み込んで特徴量エンジニアリングを実行
        # これによりELOと直近フォームの状態が更新される
        raw_df = load_raw_data(data_path)
        engineer = FeatureEngineer(initial_elo=1500.0, k_factor=32.0, recent_n=5)
        engineer.engineer_features(raw_df)

        if verbose:
            print(f"Model loaded from: {model_path}")
            print(f"Historical data loaded from: {data_path}")
            print(f"Teams with ELO ratings: {len(engineer._elo_ratings)}")

        return cls(model=model, engineer=engineer, verbose=verbose)

    def _get_team_features(self, team: str) -> dict:
        """チームの最新特徴量を取得.

        Args:
            team: チーム名

        Returns:
            特徴量の辞書
        """
        elo = self.engineer._elo_ratings.get(team, self.engineer.initial_elo)
        form = self.engineer._get_recent_form(team)

        return {
            "elo": elo,
            "recent_goals_avg": form["goals_avg"],
            "recent_conceded_avg": form["conceded_avg"],
            "recent_win_rate": form["win_rate"],
        }

    def _get_team_encoded(self, team: str) -> int:
        """チーム名をエンコードされた値に変換.

        Args:
            team: チーム名

        Returns:
            エンコードされた整数値
        """
        if self.engineer._label_encoder is None:
            raise ValueError(
                "LabelEncoderが初期化されていません。"
                "過去データで特徴量エンジニアリングを実行してください。"
            )
        # 未知のチームの場合は-1を返す
        try:
            return self.engineer._label_encoder.transform([team])[0]
        except ValueError:
            return -1

    def _prepare_features_for_match(self, match: MatchCard) -> pd.DataFrame:
        """対戦カードから特徴量DataFrameを作成.

        Args:
            match: 対戦カード

        Returns:
            特徴量のDataFrame（1行）
        """
        home_features = self._get_team_features(match.home_team)
        away_features = self._get_team_features(match.away_team)

        # NaN値は0で埋める（初回の試合など履歴がない場合）
        def fill_nan(val, default=0.0):
            return default if val is None else val

        # TotoModel.FEATURE_COLUMNSに合わせた順序でDataFrameを作成
        features_dict = {
            "home_elo": [home_features["elo"]],
            "away_elo": [away_features["elo"]],
            "home_recent_goals_avg": [fill_nan(home_features["recent_goals_avg"])],
            "home_recent_conceded_avg": [fill_nan(home_features["recent_conceded_avg"])],
            "home_recent_win_rate": [fill_nan(home_features["recent_win_rate"])],
            "away_recent_goals_avg": [fill_nan(away_features["recent_goals_avg"])],
            "away_recent_conceded_avg": [fill_nan(away_features["recent_conceded_avg"])],
            "away_recent_win_rate": [fill_nan(away_features["recent_win_rate"])],
            "home_team_encoded": [self._get_team_encoded(match.home_team)],
            "away_team_encoded": [self._get_team_encoded(match.away_team)],
        }

        return pd.DataFrame(features_dict)

    def _calculate_ev(self, probability: float, odds: float) -> float:
        """期待値を計算.

        EV = (Probability × Odds) - 1

        Args:
            probability: 予測確率
            odds: ブックメーカーオッズ

        Returns:
            期待値
        """
        return (probability * odds) - 1

    def predict_match(self, match: MatchCard) -> PredictionResult:
        """単一の試合を予測.

        Args:
            match: 対戦カード

        Returns:
            予測結果
        """
        features = self._prepare_features_for_match(match)
        probabilities = self.model.predict_proba(features)[0]

        # モデルの出力: [Away Win(0), Draw(1), Home Win(2)]
        prob_away_win = probabilities[0]
        prob_draw = probabilities[1]
        prob_home_win = probabilities[2]

        # 期待値を計算
        ev_home_win = self._calculate_ev(prob_home_win, match.odds.home_win)
        ev_draw = self._calculate_ev(prob_draw, match.odds.draw)
        ev_away_win = self._calculate_ev(prob_away_win, match.odds.away_win)

        return PredictionResult(
            match=match,
            prob_home_win=prob_home_win,
            prob_draw=prob_draw,
            prob_away_win=prob_away_win,
            ev_home_win=ev_home_win,
            ev_draw=ev_draw,
            ev_away_win=ev_away_win,
        )

    def predict_matches(self, matches: list[MatchCard]) -> list[PredictionResult]:
        """複数の試合を予測.

        Args:
            matches: 対戦カードのリスト

        Returns:
            予測結果のリスト
        """
        return [self.predict_match(match) for match in matches]

    def display_predictions(self, results: list[PredictionResult]) -> None:
        """予測結果を表形式でコンソールに出力.

        Args:
            results: 予測結果のリスト
        """
        # 英語から日本語への逆引きマップを作成
        en_to_jp_map = {}
        for jp_name, en_name in TEAM_NAME_MAP.items():
            if en_name not in en_to_jp_map:
                en_to_jp_map[en_name] = jp_name

        print("\n" + "=" * 100)
        print("【Toto予測結果】")
        print("=" * 100)

        # ヘッダー
        print(
            f"{'No.':<4} {'対戦カード':<30} "
            f"{'Home Win':>10} {'Draw':>10} {'Away Win':>10} "
            f"{'推奨':<15}"
        )
        print("-" * 100)

        for i, result in enumerate(results, 1):
            # 英語名を日本語名に変換
            home_jp = en_to_jp_map.get(result.match.home_team, result.match.home_team)
            away_jp = en_to_jp_map.get(result.match.away_team, result.match.away_team)
            match_str = f"{home_jp} vs {away_jp}"

            # 確率表示（パーセンテージ）
            home_prob_str = f"{result.prob_home_win:.1%}"
            draw_prob_str = f"{result.prob_draw:.1%}"
            away_prob_str = f"{result.prob_away_win:.1%}"

            # 推奨ベット
            best_bet = result.best_value_bet
            if best_bet:
                recommendation = f"{best_bet[0]} ★"
            else:
                recommendation = "-"

            print(
                f"{i:<4} {match_str:<30} "
                f"{home_prob_str:>10} {draw_prob_str:>10} {away_prob_str:>10} "
                f"{recommendation:<15}"
            )

        print("-" * 100)

        # 詳細な期待値表示
        print("\n【期待値 (EV) 詳細】")
        print("-" * 100)
        print(
            f"{'No.':<4} {'対戦カード':<30} "
            f"{'EV(Home)':>12} {'EV(Draw)':>12} {'EV(Away)':>12} "
            f"{'Value Bets':<20}"
        )
        print("-" * 100)

        for i, result in enumerate(results, 1):
            # 英語名を日本語名に変換
            home_jp = en_to_jp_map.get(result.match.home_team, result.match.home_team)
            away_jp = en_to_jp_map.get(result.match.away_team, result.match.away_team)
            match_str = f"{home_jp} vs {away_jp}"

            # 期待値の表示（プラスは★でハイライト）
            def format_ev(ev: float) -> str:
                if ev > 0:
                    return f"+{ev:.3f} ★"
                return f"{ev:.3f}"

            ev_home_str = format_ev(result.ev_home_win)
            ev_draw_str = format_ev(result.ev_draw)
            ev_away_str = format_ev(result.ev_away_win)

            # Value Bets
            value_bets = result.value_bets
            if value_bets:
                value_bets_str = ", ".join(
                    [f"{name}(+{ev:.2f})" for name, ev in value_bets]
                )
            else:
                value_bets_str = "なし"

            print(
                f"{i:<4} {match_str:<30} "
                f"{ev_home_str:>12} {ev_draw_str:>12} {ev_away_str:>12} "
                f"{value_bets_str:<20}"
            )

        print("-" * 100)

        # オッズ情報
        print("\n【オッズ情報】")
        print("-" * 80)
        print(
            f"{'No.':<4} {'対戦カード':<30} "
            f"{'Odds(Home)':>12} {'Odds(Draw)':>12} {'Odds(Away)':>12}"
        )
        print("-" * 80)

        for i, result in enumerate(results, 1):
            # 英語名を日本語名に変換
            home_jp = en_to_jp_map.get(result.match.home_team, result.match.home_team)
            away_jp = en_to_jp_map.get(result.match.away_team, result.match.away_team)
            match_str = f"{home_jp} vs {away_jp}"
            print(
                f"{i:<4} {match_str:<30} "
                f"{result.match.odds.home_win:>12.2f} "
                f"{result.match.odds.draw:>12.2f} "
                f"{result.match.odds.away_win:>12.2f}"
            )

        print("-" * 80)

        # サマリー
        total_value_bets = sum(len(r.value_bets) for r in results)
        matches_with_value = sum(1 for r in results if r.value_bets)

        print("\n【サマリー】")
        print(f"  対象試合数: {len(results)}")
        print(f"  Value Bet検出試合: {matches_with_value}/{len(results)}")
        print(f"  Value Bet総数: {total_value_bets}")
        print("=" * 100)


def create_sample_matches() -> list[MatchCard]:
    """サンプルの対戦カードを作成（デモ用）.

    Returns:
        対戦カードのリスト
    """
    return [
        MatchCard(
            home_team="Team_A",
            away_team="Team_B",
            odds=MatchOdds(home_win=2.10, draw=3.40, away_win=3.50),
        ),
        MatchCard(
            home_team="Team_C",
            away_team="Team_D",
            odds=MatchOdds(home_win=1.85, draw=3.60, away_win=4.20),
        ),
        MatchCard(
            home_team="Team_E",
            away_team="Team_F",
            odds=MatchOdds(home_win=2.50, draw=3.30, away_win=2.80),
        ),
        MatchCard(
            home_team="Team_G",
            away_team="Team_H",
            odds=MatchOdds(home_win=1.65, draw=3.80, away_win=5.00),
        ),
        MatchCard(
            home_team="Team_I",
            away_team="Team_J",
            odds=MatchOdds(home_win=2.20, draw=3.50, away_win=3.20),
        ),
        MatchCard(
            home_team="Team_K",
            away_team="Team_L",
            odds=MatchOdds(home_win=3.00, draw=3.40, away_win=2.30),
        ),
        MatchCard(
            home_team="Team_M",
            away_team="Team_N",
            odds=MatchOdds(home_win=1.95, draw=3.50, away_win=3.90),
        ),
        MatchCard(
            home_team="Team_O",
            away_team="Team_P",
            odds=MatchOdds(home_win=2.40, draw=3.30, away_win=2.90),
        ),
        MatchCard(
            home_team="Team_Q",
            away_team="Team_R",
            odds=MatchOdds(home_win=2.80, draw=3.40, away_win=2.50),
        ),
        MatchCard(
            home_team="Team_A",
            away_team="Team_R",
            odds=MatchOdds(home_win=2.00, draw=3.50, away_win=3.60),
        ),
    ]
