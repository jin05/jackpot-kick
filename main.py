from pathlib import Path

from src import (
    MockDataGenerator,
    FeatureEngineer,
    load_raw_data,
    save_processed_data,
    TotoModel,
    MatchPredictor,
    MatchCard,
    MatchOdds,
)
from src.predictor import create_sample_matches


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

    # ========================================
    # Phase 3: モデル学習
    # ========================================
    print("\n" + "=" * 50)
    print("Phase 3: Model Training with LightGBM...")
    print("=" * 50)

    # TotoModelの初期化
    model = TotoModel(
        n_trials=50,  # Optunaの試行回数
        n_splits=5,   # クロスバリデーションの分割数
        random_state=42,
        verbose=True,
    )

    # 特徴量とターゲットを準備
    X = model.prepare_features(processed_df)
    y = model.prepare_target(processed_df)

    print(f"\nDataset prepared:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Class distribution:")
    for cls in [0, 1, 2]:
        count = (y == cls).sum()
        print(f"    Class {cls}: {count} ({count / len(y) * 100:.1f}%)")

    # モデルの学習（ハイパーパラメータ最適化 + クロスバリデーション）
    model.fit(X, y, optimize=True)

    # 特徴量重要度の表示
    print("\nFeature importance:")
    importance_df = model.get_feature_importance()
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']}")

    # モデルの保存
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "lgbm_model.pkl"
    model.save(model_path)

    # ========================================
    # Phase 4: 予測 & 戦略
    # ========================================
    print("\n" + "=" * 50)
    print("Phase 4: Prediction & Strategy...")
    print("=" * 50)

    # MatchPredictorの初期化
    # 過去データから学習済みのFeatureEngineerを使用
    predictor = MatchPredictor(
        model=model,
        engineer=engineer,
        verbose=True,
    )

    # サンプルの対戦カードを作成
    matches = create_sample_matches()
    print(f"\n次回Toto対象試合: {len(matches)}試合")

    # 予測を実行
    results = predictor.predict_matches(matches)

    # 結果を表示
    predictor.display_predictions(results)

    print("\n" + "=" * 50)
    print("All phases completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
