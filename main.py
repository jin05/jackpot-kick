from pathlib import Path

from src import MockDataGenerator, FeatureEngineer, load_raw_data, save_processed_data


def main():
    # ========================================
    # Phase 1: データ生成
    # ========================================
    print("=" * 50)
    print("Phase 1: Generating mock match data...")
    print("=" * 50)

    generator = MockDataGenerator(seed=42)
    df = generator.generate_historical_data(n_samples=1000)

    # 保存先ディレクトリの作成
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # CSVファイルとして保存
    raw_output_path = data_dir / "raw_match_data.csv"
    df.to_csv(raw_output_path, index=False)

    print(f"Generated {len(df)} match records")
    print(f"Saved to: {raw_output_path}")
    print(f"\nSample raw data:")
    print(df.head(5))

    # ========================================
    # Phase 2: 特徴量エンジニアリング
    # ========================================
    print("\n" + "=" * 50)
    print("Phase 2: Feature Engineering...")
    print("=" * 50)

    # 生データを読み込み
    raw_df = load_raw_data(raw_output_path)

    # 特徴量エンジニアリングを実行
    engineer = FeatureEngineer(
        initial_elo=1500.0,
        k_factor=32.0,
        recent_n=5,
    )
    processed_df = engineer.engineer_features(raw_df)

    # 処理済みデータを保存
    processed_output_path = data_dir / "processed_features.csv"
    save_processed_data(processed_df, processed_output_path)

    # 結果を表示
    print(f"\nAdded features:")
    new_columns = [
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
    for col in new_columns:
        print(f"  - {col}")

    print(f"\nSample processed data:")
    print(processed_df.head(10))

    print(f"\nDataFrame shape: {processed_df.shape}")
    print(f"\nColumns: {list(processed_df.columns)}")

    # チームラベルのマッピングを表示
    print(f"\nTeam label mapping:")
    for encoded, name in engineer.get_team_labels().items():
        print(f"  {encoded}: {name}")


if __name__ == "__main__":
    main()
