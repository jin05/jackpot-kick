from pathlib import Path

from src import MockDataGenerator


def main():
    # データ生成
    generator = MockDataGenerator(seed=42)
    df = generator.generate_historical_data(n_samples=1000)

    # 保存先ディレクトリの作成
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # CSVファイルとして保存
    output_path = data_dir / "raw_match_data.csv"
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} match records")
    print(f"Saved to: {output_path}")
    print(f"\nSample data:")
    print(df.head(10))


if __name__ == "__main__":
    main()
