import random
from datetime import datetime, timedelta

import pandas as pd


class MockDataGenerator:
    """Jリーグ風のダミー対戦データを生成するクラス"""

    TEAMS = [f"Team_{chr(65 + i)}" for i in range(18)]  # Team_A ~ Team_R

    def __init__(self, seed: int | None = None):
        """
        Args:
            seed: 乱数シード（再現性のため）
        """
        if seed is not None:
            random.seed(seed)

    def generate_historical_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Jリーグ風のダミー対戦データを生成する

        Args:
            n_samples: 生成するサンプル数

        Returns:
            以下のカラムを持つDataFrame:
            - date: 日付
            - home_team: ホームチーム名
            - away_team: アウェイチーム名
            - home_score: ホームチームの得点 (0-5)
            - away_score: アウェイチームの得点 (0-5)
            - result: 試合結果 (0=Away Win, 1=Draw, 2=Home Win)
        """
        data = []
        base_date = datetime(2020, 1, 1)

        for i in range(n_samples):
            # ランダムな日付を生成（2020年から約4年分）
            date = base_date + timedelta(days=random.randint(0, 1460))

            # ホームとアウェイのチームをランダムに選択（同じチーム同士は除外）
            home_team = random.choice(self.TEAMS)
            away_team = random.choice([t for t in self.TEAMS if t != home_team])

            # ランダムな得点を生成 (0-5)
            home_score = random.randint(0, 5)
            away_score = random.randint(0, 5)

            # 結果を得点から算出
            if home_score > away_score:
                result = 2  # Home Win
            elif home_score < away_score:
                result = 0  # Away Win
            else:
                result = 1  # Draw

            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "result": result,
            })

        df = pd.DataFrame(data)
        df = df.sort_values("date").reset_index(drop=True)
        return df
