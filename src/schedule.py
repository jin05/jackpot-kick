"""
MatchScheduleFetcher - Jリーグの次節対戦カードを自動取得するクラス

Phase 5: 試合スケジュールの自動取得
"""

import re
import time
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from .scraper import TEAM_NAME_MAP
from .predictor import MatchCard, MatchOdds


@dataclass
class ScheduledMatch:
    """スケジュールされた試合情報を表すデータクラス.

    Attributes:
        date: 試合日（YYYY-MM-DD形式）
        kickoff_time: キックオフ時刻（HH:MM形式、取得できない場合はNone）
        home_team: ホームチーム名（日本語）
        away_team: アウェイチーム名（日本語）
        home_team_en: ホームチーム名（英語、変換後）
        away_team_en: アウェイチーム名（英語、変換後）
        stadium: スタジアム名（取得できる場合）
        matchday: 節番号（取得できる場合）
    """

    date: str
    kickoff_time: str | None
    home_team: str
    away_team: str
    home_team_en: str
    away_team_en: str
    stadium: str | None = None
    matchday: int | None = None


# デフォルトのオッズ設定
DEFAULT_ODDS = MatchOdds(
    home_win=2.5,
    draw=3.0,
    away_win=2.5,
)


class MatchScheduleFetcher:
    """
    Jリーグ公式サイトから次節の対戦カードを取得するクラス

    Attributes:
        base_url: JリーグスケジュールページのベースURL
        sleep_time: リクエスト間の待機時間（秒）
        verbose: デバッグ出力を行うかどうか
    """

    # Jリーグ公式サイトのスケジュールページURL
    BASE_URL = "https://www.jleague.jp/match/search/j1/"

    def __init__(
        self,
        sleep_time: float = 1.0,
        verbose: bool = True,
    ):
        """
        Args:
            sleep_time: リクエスト間の待機時間（秒）。デフォルトは1.0秒
            verbose: デバッグ出力を行うかどうか
        """
        self.sleep_time = max(1.0, sleep_time)
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        })

    def _log(self, message: str) -> None:
        """ログ出力"""
        if self.verbose:
            print(f"[MatchScheduleFetcher] {message}")

    def _normalize_team_name(self, team_name: str) -> str:
        """
        日本語チーム名を英語表記に正規化

        Args:
            team_name: 日本語のチーム名

        Returns:
            英語表記のチーム名。マッピングがない場合は元の名前を返す
        """
        # 空白・全角文字を正規化
        normalized = team_name.strip()
        # 全角スペースを半角に
        normalized = normalized.replace("\u3000", " ")
        # 前後の空白を削除
        normalized = normalized.strip()

        if normalized in TEAM_NAME_MAP:
            return TEAM_NAME_MAP[normalized]

        # 部分一致を試みる
        for jp_name, en_name in TEAM_NAME_MAP.items():
            if jp_name in normalized or normalized in jp_name:
                return en_name

        # マッピングがない場合は警告を出して元の名前を返す
        self._log(f"Warning: Team name '{team_name}' not found in mapping")
        return normalized

    def fetch_next_matches(self) -> list[ScheduledMatch]:
        """
        次節の対戦カードを取得

        Returns:
            次節の対戦カードのリスト
        """
        self._log("Fetching next J1 League matches...")

        matches = []

        try:
            time.sleep(self.sleep_time)

            response = self.session.get(
                self.BASE_URL,
                timeout=30,
            )
            response.raise_for_status()
            response.encoding = "utf-8"

            soup = BeautifulSoup(response.text, "html.parser")

            # Jリーグ公式サイトの構造に基づいてパース
            # 試合情報を含むセクションを探す
            matches = self._parse_jleague_schedule(soup)

            if not matches:
                self._log("Warning: No matches found from J.League site. Trying alternative...")
                # フォールバック: 別のアプローチを試す
                matches = self._parse_alternative_structure(soup)

            # 重複を削除
            seen = set()
            unique_matches = []
            for match in matches:
                key = (match.home_team_en, match.away_team_en)
                if key not in seen:
                    seen.add(key)
                    unique_matches.append(match)
            matches = unique_matches

            self._log(f"Found {len(matches)} scheduled matches")

        except requests.RequestException as e:
            self._log(f"Error fetching schedule: {e}")

        return matches

    def _parse_jleague_schedule(self, soup: BeautifulSoup) -> list[ScheduledMatch]:
        """
        Jリーグ公式サイトのスケジュールページをパース

        Args:
            soup: BeautifulSoupオブジェクト

        Returns:
            対戦カードのリスト
        """
        matches = []

        # 試合一覧を探す（複数のセレクタを試す）
        # パターン1: matchListセクション
        match_sections = soup.select(".matchList-match, .match-card, .matchCard")

        if not match_sections:
            # パターン2: テーブル形式
            match_sections = soup.select("table.matchTable tbody tr, .match-row")

        if not match_sections:
            # パターン3: リスト形式
            match_sections = soup.select(".schedule-list li, .scheduleList li")

        if not match_sections:
            # パターン4: 一般的なdiv構造
            match_sections = soup.select("[class*='match'], [class*='Match']")

        current_date = None
        current_matchday = None

        for section in match_sections:
            try:
                match = self._extract_match_info(section, current_date, current_matchday)
                if match:
                    matches.append(match)
                    current_date = match.date  # 日付を更新
            except Exception as e:
                self._log(f"Error parsing match section: {e}")
                continue

        # 日付ヘッダーも探して日付を取得
        date_headers = soup.select(".date-header, .matchDay, [class*='date']")
        for header in date_headers:
            date_text = header.get_text(strip=True)
            parsed_date = self._parse_date(date_text)
            if parsed_date:
                current_date = parsed_date

        return matches

    def _extract_match_info(
        self,
        element: BeautifulSoup,
        default_date: str | None,
        default_matchday: int | None,
    ) -> ScheduledMatch | None:
        """
        HTML要素から試合情報を抽出

        Args:
            element: BeautifulSoupの要素
            default_date: デフォルトの日付
            default_matchday: デフォルトの節番号

        Returns:
            ScheduledMatchオブジェクト、または抽出に失敗した場合はNone
        """
        text = element.get_text(" ", strip=True)

        # チーム名を探す（vs, VS, ー, - などで区切られている）
        # パターン: "ホームチーム vs アウェイチーム" または "ホームチーム - アウェイチーム"
        team_pattern = r"([^\s\d\-:]+(?:\s+[^\s\d\-:]+)?)\s*(?:vs|VS|ＶＳ|ー|\-|−)\s*([^\s\d\-:]+(?:\s+[^\s\d\-:]+)?)"
        team_match = re.search(team_pattern, text)

        if not team_match:
            # 別パターン: ホームとアウェイが別の要素にある場合
            home_elem = element.select_one(".home-team, .homeTeam, [class*='home']")
            away_elem = element.select_one(".away-team, .awayTeam, [class*='away']")

            if home_elem and away_elem:
                home_team = home_elem.get_text(strip=True)
                away_team = away_elem.get_text(strip=True)
            else:
                return None
        else:
            home_team = team_match.group(1).strip()
            away_team = team_match.group(2).strip()

        # 日付を探す
        date_elem = element.select_one(".date, [class*='date']")
        match_date = default_date
        if date_elem:
            date_text = date_elem.get_text(strip=True)
            parsed_date = self._parse_date(date_text)
            if parsed_date:
                match_date = parsed_date

        # 日付がまだない場合はテキスト全体から探す
        if not match_date:
            parsed_date = self._parse_date(text)
            if parsed_date:
                match_date = parsed_date

        # デフォルト日付を設定（見つからない場合は今日）
        if not match_date:
            match_date = datetime.now().strftime("%Y-%m-%d")

        # キックオフ時刻を探す
        kickoff_time = None
        time_pattern = r"(\d{1,2}:\d{2})"
        time_match = re.search(time_pattern, text)
        if time_match:
            kickoff_time = time_match.group(1)

        # スタジアムを探す
        stadium_elem = element.select_one(".stadium, [class*='stadium'], [class*='venue']")
        stadium = stadium_elem.get_text(strip=True) if stadium_elem else None

        # チーム名を英語に変換
        home_team_en = self._normalize_team_name(home_team)
        away_team_en = self._normalize_team_name(away_team)

        # 不正なチーム名をフィルタリング（数字のみ、空文字など）
        if not home_team or not away_team:
            return None
        if home_team.isdigit() or away_team.isdigit():
            return None

        # 同じチーム同士の対戦をスキップ（パースエラー）
        if home_team_en == away_team_en:
            return None

        # TEAM_NAME_MAPに存在しないチームをスキップ（J2/J3チームなど）
        valid_teams = set(TEAM_NAME_MAP.values())
        if home_team_en not in valid_teams or away_team_en not in valid_teams:
            return None

        return ScheduledMatch(
            date=match_date,
            kickoff_time=kickoff_time,
            home_team=home_team,
            away_team=away_team,
            home_team_en=home_team_en,
            away_team_en=away_team_en,
            stadium=stadium,
            matchday=default_matchday,
        )

    def _parse_alternative_structure(self, soup: BeautifulSoup) -> list[ScheduledMatch]:
        """
        代替パース方法: ページ全体のテキストからパターンマッチング

        Args:
            soup: BeautifulSoupオブジェクト

        Returns:
            対戦カードのリスト
        """
        matches = []
        text = soup.get_text()

        # "チームA vs チームB" または "チームA - チームB" パターンを探す
        pattern = r"([^\n\d]+?)\s*(?:vs|VS|ＶＳ)\s*([^\n\d]+)"
        found_matches = re.findall(pattern, text)

        for home, away in found_matches:
            home = home.strip()
            away = away.strip()

            # 明らかに無効なマッチをスキップ
            if len(home) < 2 or len(away) < 2:
                continue
            if any(c in home for c in ["[", "]", "（", "）", "(", ")"]):
                # カッコ内の文字を削除
                home = re.sub(r"[\[\]（）()].*", "", home).strip()
            if any(c in away for c in ["[", "]", "（", "）", "(", ")"]):
                away = re.sub(r"[\[\]（）()].*", "", away).strip()

            if not home or not away:
                continue

            home_en = self._normalize_team_name(home)
            away_en = self._normalize_team_name(away)

            # 変換後も同じ（マッピングなし）場合は有効なチームでない可能性
            # ただし、既存のTEAM_NAME_MAPにあるチームのみを追加
            if home_en in TEAM_NAME_MAP.values() or away_en in TEAM_NAME_MAP.values():
                matches.append(ScheduledMatch(
                    date=datetime.now().strftime("%Y-%m-%d"),
                    kickoff_time=None,
                    home_team=home,
                    away_team=away,
                    home_team_en=home_en,
                    away_team_en=away_en,
                ))

        # 重複を削除
        seen = set()
        unique_matches = []
        for match in matches:
            key = (match.home_team_en, match.away_team_en)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)

        return unique_matches

    def _parse_date(self, text: str) -> str | None:
        """
        テキストから日付を抽出

        Args:
            text: 日付を含む可能性のあるテキスト

        Returns:
            YYYY-MM-DD形式の日付文字列、または抽出に失敗した場合はNone
        """
        # パターン1: 2024年3月1日
        pattern1 = r"(\d{4})年(\d{1,2})月(\d{1,2})日"
        match1 = re.search(pattern1, text)
        if match1:
            year, month, day = match1.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"

        # パターン2: 2024/3/1 または 2024-03-01
        pattern2 = r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})"
        match2 = re.search(pattern2, text)
        if match2:
            year, month, day = match2.groups()
            return f"{year}-{int(month):02d}-{int(day):02d}"

        # パターン3: 3/1 または 3月1日（年なし、現在の年を使用）
        pattern3 = r"(\d{1,2})[/月](\d{1,2})"
        match3 = re.search(pattern3, text)
        if match3:
            month, day = match3.groups()
            year = datetime.now().year
            return f"{year}-{int(month):02d}-{int(day):02d}"

        return None

    def get_match_cards_with_default_odds(
        self,
        matches: list[ScheduledMatch] | None = None,
    ) -> list[MatchCard]:
        """
        対戦カードにデフォルトオッズを付与してMatchCardリストを生成

        Args:
            matches: ScheduledMatchのリスト（Noneの場合は自動取得）

        Returns:
            MatchCardのリスト（重複は削除済み）
        """
        if matches is None:
            matches = self.fetch_next_matches()

        # 重複を削除
        seen = set()
        unique_matches = []
        for match in matches:
            key = (match.home_team_en, match.away_team_en)
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)

        match_cards = []
        for match in unique_matches:
            card = MatchCard(
                home_team=match.home_team_en,
                away_team=match.away_team_en,
                odds=DEFAULT_ODDS,
            )
            match_cards.append(card)

        return match_cards

    def display_schedule(self, matches: list[ScheduledMatch]) -> None:
        """
        取得したスケジュールをコンソールに表示

        Args:
            matches: ScheduledMatchのリスト
        """
        print("\n" + "=" * 80)
        print("【次節 J1リーグ 対戦カード】")
        print("=" * 80)

        if not matches:
            print("  対戦カードが取得できませんでした。")
            print("  ※ サイト構造が変更された可能性があります。")
            return

        print(f"{'No.':<4} {'日付':<12} {'時刻':<8} {'ホーム':^15} {'vs':^4} {'アウェイ':^15}")
        print("-" * 80)

        for i, match in enumerate(matches, 1):
            kickoff = match.kickoff_time or "--:--"
            print(
                f"{i:<4} {match.date:<12} {kickoff:<8} "
                f"{match.home_team_en:^15} {'vs':^4} {match.away_team_en:^15}"
            )

        print("-" * 80)
        print(f"  取得試合数: {len(matches)}")
        print("=" * 80)

    def display_match_cards_with_odds(self, match_cards: list[MatchCard]) -> None:
        """
        オッズ付き対戦カードをコンソールに表示

        Args:
            match_cards: MatchCardのリスト
        """
        print("\n" + "=" * 100)
        print("【対戦カード & 仮オッズ】")
        print("=" * 100)

        if not match_cards:
            print("  対戦カードがありません。")
            return

        print(
            f"{'No.':<4} {'ホーム':^18} {'vs':^4} {'アウェイ':^18} "
            f"{'Home Win':>10} {'Draw':>10} {'Away Win':>10}"
        )
        print("-" * 100)

        for i, card in enumerate(match_cards, 1):
            print(
                f"{i:<4} {card.home_team:^18} {'vs':^4} {card.away_team:^18} "
                f"{card.odds.home_win:>10.2f} {card.odds.draw:>10.2f} {card.odds.away_win:>10.2f}"
            )

        print("-" * 100)
        print(f"  ※ オッズはデフォルト値（Home: {DEFAULT_ODDS.home_win}, Draw: {DEFAULT_ODDS.draw}, Away: {DEFAULT_ODDS.away_win}）")
        print("  ※ 実際のオッズに変更する場合は手動で更新してください。")
        print("=" * 100)


def fetch_next_round_matches(verbose: bool = True) -> list[MatchCard]:
    """
    次節の対戦カードを取得するユーティリティ関数

    Args:
        verbose: 詳細出力の有無

    Returns:
        MatchCardのリスト（デフォルトオッズ付き）
    """
    fetcher = MatchScheduleFetcher(verbose=verbose)
    return fetcher.get_match_cards_with_default_odds()


# サンプル対戦カード（スクレイピング失敗時のフォールバック用）
SAMPLE_J1_MATCHES = [
    ScheduledMatch(
        date=datetime.now().strftime("%Y-%m-%d"),
        kickoff_time="14:00",
        home_team="川崎フロンターレ",
        away_team="横浜F・マリノス",
        home_team_en="Kawasaki Frontale",
        away_team_en="Yokohama FM",
    ),
    ScheduledMatch(
        date=datetime.now().strftime("%Y-%m-%d"),
        kickoff_time="14:00",
        home_team="浦和レッズ",
        away_team="鹿島アントラーズ",
        home_team_en="Urawa Reds",
        away_team_en="Kashima Antlers",
    ),
    ScheduledMatch(
        date=datetime.now().strftime("%Y-%m-%d"),
        kickoff_time="14:00",
        home_team="ヴィッセル神戸",
        away_team="サンフレッチェ広島",
        home_team_en="Vissel Kobe",
        away_team_en="Sanfrecce Hiroshima",
    ),
    ScheduledMatch(
        date=datetime.now().strftime("%Y-%m-%d"),
        kickoff_time="14:00",
        home_team="ガンバ大阪",
        away_team="セレッソ大阪",
        home_team_en="Gamba Osaka",
        away_team_en="Cerezo Osaka",
    ),
    ScheduledMatch(
        date=datetime.now().strftime("%Y-%m-%d"),
        kickoff_time="14:00",
        home_team="FC東京",
        away_team="名古屋グランパス",
        home_team_en="FC Tokyo",
        away_team_en="Nagoya Grampus",
    ),
]
