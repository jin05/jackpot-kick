"""
JLeagueScraper - Jリーグ公式データサイトから試合結果をスクレイピングするクラス
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


# J1リーグチーム名マッピング（日本語表記 → 英語表記）
# 2023-2025シーズンのチームを網羅
TEAM_NAME_MAP: dict[str, str] = {
    # 北海道・東北
    "北海道コンサドーレ札幌": "Consadole Sapporo",
    "コンサドーレ札幌": "Consadole Sapporo",
    "札幌": "Consadole Sapporo",
    # 関東
    "鹿島アントラーズ": "Kashima Antlers",
    "鹿島": "Kashima Antlers",
    "浦和レッズ": "Urawa Reds",
    "浦和レッドダイヤモンズ": "Urawa Reds",
    "浦和": "Urawa Reds",
    "柏レイソル": "Kashiwa Reysol",
    "柏": "Kashiwa Reysol",
    "FC東京": "FC Tokyo",
    "東京": "FC Tokyo",
    "川崎フロンターレ": "Kawasaki Frontale",
    "川崎F": "Kawasaki Frontale",
    "川崎": "Kawasaki Frontale",
    "横浜F・マリノス": "Yokohama FM",
    "横浜Ｆ・マリノス": "Yokohama FM",
    "横浜FM": "Yokohama FM",
    "横浜マリノス": "Yokohama FM",
    "マリノス": "Yokohama FM",
    "横浜FC": "Yokohama FC",
    "湘南ベルマーレ": "Shonan Bellmare",
    "湘南": "Shonan Bellmare",
    # 中部
    "アルビレックス新潟": "Albirex Niigata",
    "新潟": "Albirex Niigata",
    "清水エスパルス": "Shimizu S-Pulse",
    "清水": "Shimizu S-Pulse",
    "ジュビロ磐田": "Jubilo Iwata",
    "磐田": "Jubilo Iwata",
    "名古屋グランパス": "Nagoya Grampus",
    "名古屋グランパスエイト": "Nagoya Grampus",
    "名古屋": "Nagoya Grampus",
    # 関西
    "京都サンガF.C.": "Kyoto Sanga",
    "京都サンガFC": "Kyoto Sanga",
    "京都サンガＦ.Ｃ.": "Kyoto Sanga",
    "京都": "Kyoto Sanga",
    "ガンバ大阪": "Gamba Osaka",
    "G大阪": "Gamba Osaka",
    "Ｇ大阪": "Gamba Osaka",
    "セレッソ大阪": "Cerezo Osaka",
    "C大阪": "Cerezo Osaka",
    "Ｃ大阪": "Cerezo Osaka",
    "ヴィッセル神戸": "Vissel Kobe",
    "神戸": "Vissel Kobe",
    # 中国・四国
    "サンフレッチェ広島": "Sanfrecce Hiroshima",
    "広島": "Sanfrecce Hiroshima",
    # 九州
    "アビスパ福岡": "Avispa Fukuoka",
    "福岡": "Avispa Fukuoka",
    "サガン鳥栖": "Sagan Tosu",
    "鳥栖": "Sagan Tosu",
    # 2024年以降J1参入チーム
    "東京ヴェルディ": "Tokyo Verdy",
    "東京V": "Tokyo Verdy",
    "ヴェルディ": "Tokyo Verdy",
    "町田ゼルビア": "Machida Zelvia",
    "FC町田ゼルビア": "Machida Zelvia",
    "町田": "Machida Zelvia",
}


class JLeagueScraper:
    """
    Jリーグ公式データサイトから試合結果をスクレイピングするクラス

    Attributes:
        base_url: JリーグデータサイトのベースURL
        sleep_time: リクエスト間の待機時間（秒）
        verbose: デバッグ出力を行うかどうか
    """

    # JリーグデータサイトのURL
    BASE_URL = "https://data.j-league.or.jp/SFMS01/search"

    def __init__(
        self,
        sleep_time: float = 1.5,
        verbose: bool = True,
    ):
        """
        Args:
            sleep_time: リクエスト間の待機時間（秒）。デフォルトは1.5秒
            verbose: デバッグ出力を行うかどうか
        """
        self.sleep_time = max(1.0, sleep_time)  # 最低1秒は待機
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        })

    def _log(self, message: str) -> None:
        """ログ出力"""
        if self.verbose:
            print(f"[JLeagueScraper] {message}")

    def _normalize_team_name(self, team_name: str) -> str:
        """
        日本語チーム名を英語表記に正規化

        Args:
            team_name: 日本語のチーム名

        Returns:
            英語表記のチーム名。マッピングがない場合は元の名前を返す
        """
        # 空白を削除してマッチングを試みる
        normalized = team_name.strip()
        if normalized in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[normalized]

        # 部分一致を試みる
        for jp_name, en_name in TEAM_NAME_MAP.items():
            if jp_name in normalized or normalized in jp_name:
                return en_name

        # マッピングがない場合は警告を出して元の名前を返す
        self._log(f"Warning: Team name '{team_name}' not found in mapping")
        return normalized

    def _calculate_result(self, home_score: int, away_score: int) -> int:
        """
        試合結果を算出

        Args:
            home_score: ホームチームの得点
            away_score: アウェイチームの得点

        Returns:
            0: Away Win, 1: Draw, 2: Home Win
        """
        if home_score > away_score:
            return 2  # Home Win
        elif home_score < away_score:
            return 0  # Away Win
        else:
            return 1  # Draw

    def _convert_date_with_year_context(self, date_str: str, target_year: int) -> str:
        """
        年度パラメータのコンテキストを使用して、短縮形の年号を4桁に修正

        Jリーグサイトでは年末から翌年の日付が「24/12/31」と「25/01/10」のように
        2桁年号で表記されます。このメソッドは、取得対象の年度を基準に正確に
        年号を補正します。

        例：
        - target_year=2025, date_str="25/01/10" → "2025-01-10"
        - target_year=2025, date_str="24/12/31" → "2024-12-31"
        - target_year=2024, date_str="24/02/23" → "2024-02-23"

        Args:
            date_str: "YY/MM/DD"形式の日付文字列（例: "25/01/10"）
            target_year: Jリーグシーズンの年度（例: 2025）

        Returns:
            "YYYY-MM-DD"形式の日付文字列
        """
        try:
            # まず2桁年号をパースして仮の日付を作成
            date_obj = datetime.strptime(date_str, "%y/%m/%d")
            parsed_year_short = date_obj.year  # %yで解釈された年（2000-2099）

            # target_yearを基準に正確な年を決定
            # Jリーグシーズンは1月から12月。例：2025年シーズンは2025/1-12月の試合
            # 一方、翌シーズンの試合は翌年になる（2026年）

            # パースされた年が対象年度と一致する場合
            if parsed_year_short == target_year:
                result_year = target_year
            # パースされた年が前年（シーズン開始前の12月）の場合
            elif parsed_year_short == target_year - 1:
                result_year = target_year - 1
            # パースされた年が翌年（シーズン終了後の1月など）の場合
            elif parsed_year_short == target_year + 1:
                result_year = target_year + 1
            # 上記以外は、パースされた年をそのまま使用
            else:
                # 年末年始の処理：12月→前年、1月→当該年として判定
                month = date_obj.month
                if month == 12:
                    result_year = target_year - 1
                elif month == 1:
                    result_year = target_year
                else:
                    result_year = target_year

            # 正確な年で日付を再構成
            corrected_date = date_obj.replace(year=result_year)
            return corrected_date.strftime("%Y-%m-%d")

        except ValueError:
            return None

    def fetch_match_results(
        self,
        year: int,
        competition: str = "J1",
    ) -> pd.DataFrame:
        """
        指定した年度のリーグ試合結果を取得

        Args:
            year: 取得する年度（例: 2023, 2024）
            competition: リーグ種別（"J1", "J2", "J3"）

        Returns:
            以下のカラムを持つDataFrame:
            - date: YYYY-MM-DD形式の日付
            - home_team: ホームチーム名（英語表記）
            - away_team: アウェイチーム名（英語表記）
            - home_score: ホームチームの得点
            - away_score: アウェイチームの得点
            - result: 試合結果 (0=Away Win, 1=Draw, 2=Home Win)
        """
        self._log(f"Fetching {competition} {year} match results...")

        matches = []

        # JリーグデータサイトへのPOSTパラメータ
        # 注意: 実際のサイト構造に応じて調整が必要な場合があります
        competition_codes = {
            "J1": "1",
            "J2": "2",
            "J3": "3",
        }

        params = {
            "competition_years": str(year),
            "competition_frame_ids": competition_codes.get(competition, "1"),
            "tv_relay_station_name": "",
        }

        try:
            # サーバー負荷軽減のため待機
            time.sleep(self.sleep_time)

            response = self.session.post(
                self.BASE_URL,
                data=params,
                timeout=30,
            )
            response.raise_for_status()
            response.encoding = "utf-8"

            soup = BeautifulSoup(response.text, "html.parser")

            # 試合結果テーブルを探す
            # JリーグデータサイトのHTML構造に基づいてセレクタを指定
            match_tables = soup.select("table.search-table tbody tr")

            if not match_tables:
                # 代替セレクタを試す
                self._log("Warning: table.search-table not found, trying alternative selectors")
                match_tables = soup.select("table tbody tr")

            for row in match_tables:
                try:
                    cells = row.find_all("td")
                    if len(cells) < 8:  # 最低8セル必要（日付、ホーム、スコア、アウェイを含む）
                        continue

                    # HTML構造に応じてデータを抽出
                    # 実際の構造: [0]シーズン, [1]大会, [2]節, [3]試合日, [4]KO時刻,
                    #             [5]ホーム, [6]スコア, [7]アウェイ, [8]スタジアム, ...
                    date_text = cells[3].get_text(strip=True)  # 試合日
                    home_team_text = cells[5].get_text(strip=True)  # ホーム
                    score_text = cells[6].get_text(strip=True)  # スコア
                    away_team_text = cells[7].get_text(strip=True)  # アウェイ

                    # 日付のパース
                    try:
                        # 複数の日付フォーマットに対応
                        date_str = None
                        # まず "YY/MM/DD(曜日)" 形式をクリーンアップ
                        date_clean = date_text.split("(")[0].strip()  # "24/02/23(金・祝)" -> "24/02/23"

                        # 年度パラメータのコンテキストを使用して日付を補正
                        # これにより、年末から翌年にかけての日付誤認識を防止
                        if "/" in date_clean and date_clean.count("/") == 2:
                            # "YY/MM/DD" 形式
                            date_str = self._convert_date_with_year_context(date_clean, year)
                        else:
                            # その他のフォーマットに対応
                            date_obj = None
                            for fmt in ["%Y/%m/%d", "%Y-%m-%d", "%Y年%m月%d日"]:
                                try:
                                    date_obj = datetime.strptime(date_clean, fmt)
                                    date_str = date_obj.strftime("%Y-%m-%d")
                                    break
                                except ValueError:
                                    continue

                        if date_str is None:
                            continue
                    except ValueError:
                        continue

                    # スコアのパース（例: "2-1", "2 - 1", "2−1"）
                    score_text = score_text.replace(" ", "").replace("−", "-")
                    if "-" not in score_text:
                        continue

                    score_parts = score_text.split("-")
                    if len(score_parts) != 2:
                        continue

                    try:
                        home_score = int(score_parts[0])
                        away_score = int(score_parts[1])
                    except ValueError:
                        continue

                    # チーム名の正規化
                    home_team = self._normalize_team_name(home_team_text)
                    away_team = self._normalize_team_name(away_team_text)

                    # 結果を算出
                    result = self._calculate_result(home_score, away_score)

                    matches.append({
                        "date": date_str,
                        "home_team": home_team,
                        "away_team": away_team,
                        "home_score": home_score,
                        "away_score": away_score,
                        "result": result,
                    })

                except (IndexError, AttributeError) as e:
                    self._log(f"Error parsing row: {e}")
                    continue

            self._log(f"Found {len(matches)} matches for {competition} {year}")

        except requests.RequestException as e:
            self._log(f"Error fetching data: {e}")
            return pd.DataFrame(columns=[
                "date", "home_team", "away_team",
                "home_score", "away_score", "result"
            ])

        df = pd.DataFrame(matches)
        if not df.empty:
            df = df.sort_values("date").reset_index(drop=True)

        return df

    def fetch_multiple_years(
        self,
        years: list[int],
        competition: str = "J1",
    ) -> pd.DataFrame:
        """
        複数年度の試合結果を取得

        Args:
            years: 取得する年度のリスト（例: [2023, 2024]）
            competition: リーグ種別

        Returns:
            全年度の試合結果を結合したDataFrame
        """
        all_matches = []

        for year in years:
            df = self.fetch_match_results(year, competition)
            if not df.empty:
                all_matches.append(df)

            # 年度間でも待機
            if year != years[-1]:
                time.sleep(self.sleep_time)

        if not all_matches:
            return pd.DataFrame(columns=[
                "date", "home_team", "away_team",
                "home_score", "away_score", "result"
            ])

        combined_df = pd.concat(all_matches, ignore_index=True)
        combined_df = combined_df.sort_values("date").reset_index(drop=True)

        self._log(f"Total matches fetched: {len(combined_df)}")
        return combined_df

    def save_to_csv(
        self,
        df: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
        """
        DataFrameをCSVファイルとして保存

        Args:
            df: 保存するDataFrame
            output_path: 出力先のパス
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        self._log(f"Saved {len(df)} matches to {output_path}")


def scrape_jleague_data(
    years: list[int] | None = None,
    output_path: str | Path = "data/raw_match_data.csv",
    competition: str = "J1",
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Jリーグの試合データをスクレイピングして保存するユーティリティ関数

    Args:
        years: 取得する年度のリスト。Noneの場合は過去3年分
        output_path: 出力先のCSVパス
        competition: リーグ種別
        force_download: Trueの場合、既存ファイルがあっても再ダウンロード

    Returns:
        試合結果のDataFrame
    """
    output_path = Path(output_path)

    # 既にファイルが存在する場合はスキップ
    if output_path.exists() and not force_download:
        print(f"[JLeagueScraper] File already exists: {output_path}")
        print("[JLeagueScraper] Loading from existing file...")
        return pd.read_csv(output_path)

    # デフォルトの年度を設定
    if years is None:
        current_year = datetime.now().year
        years = [current_year - 2, current_year - 1, current_year]

    scraper = JLeagueScraper()
    df = scraper.fetch_multiple_years(years, competition)

    if not df.empty:
        scraper.save_to_csv(df, output_path)

    return df
