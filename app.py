#----------streamlitでのUI実装--------------
from __future__ import annotations

import base64
import json
import itertools
import os
import re
import subprocess
import sys
import time
import unicodedata
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import requests
import streamlit as st
import streamlit.components.v1 as components
from bs4 import BeautifulSoup

import io_json
import market_prob
import tickets as ticket_probs
import google_sheets
from ev import (
    compute_ev,
    compute_ev_prime,
    compute_ev_prime_range,
    compute_ev_range,
    compute_kelly_fraction_clipped,
    compute_kelly_range,
    compute_p_log_o,
    compute_p_log_o_range,
)


# ----------------------------
# Fetch
# ----------------------------
_META_CHARSET_RE = re.compile(
    r"""charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)\s*["']?""", re.IGNORECASE
)


def _normalize_encoding(name: str) -> str:
    n = name.strip().lower().replace("_", "-")
    # netkeiba系でよく出る表記ゆれをPython codec名に寄せる
    if n in ("shift-jis", "shiftjis", "sjis", "x-sjis", "windows-31j"):
        return "cp932"
    if n in ("euc-jp", "eucjp"):
        return "euc_jp"
    if n in ("utf8", "utf-8"):
        return "utf-8"
    return name


def _detect_html_encoding(
    content: bytes,
    header_encoding: Optional[str],
    apparent_encoding: Optional[str],
) -> str:
    # requests の r.encoding は header が無いと ISO-8859-1 になりがちで、
    # 日本語ページだとそれが原因で "¥ô¥©..." のように文字化けします。
    if header_encoding and header_encoding.lower() not in ("iso-8859-1", "latin-1"):
        return _normalize_encoding(header_encoding)

    # HTML中の <meta charset=...> を優先
    head = content[:4096].decode("ascii", errors="ignore")
    m = _META_CHARSET_RE.search(head)
    if m:
        return _normalize_encoding(m.group(1))

    # 最後に推定（charset_normalizer / chardet）を使う
    if apparent_encoding:
        return _normalize_encoding(apparent_encoding)

    return "utf-8"


def fetch_html(url: str, timeout: int = 15) -> str:
    # netkeiba 系は UA が弱いとブロック/簡易ページになることがあるため、ブラウザ寄せで送る
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }
    # ネットワークが不安定なときの一時的エラーを吸収する（403/429は除外）
    retries = 2
    backoff_sec = 0.8
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)

            if r.status_code in (403, 429):
                raise RuntimeError(
                    f"Fetch blocked (status={r.status_code}). Do not retry repeatedly; wait and try later."
                )

            if r.status_code in (408, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff_sec * (attempt + 1))
                continue

            r.raise_for_status()
            enc = _detect_html_encoding(r.content, r.encoding, r.apparent_encoding)
            html = r.content.decode(enc, errors="replace")

            # 200 でも「ブロック/JS必須/エラーページ」が返ることがあるので簡易検知
            t = unicodedata.normalize("NFKC", BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
            blocked_words = [
                "Access Denied",
                "Forbidden",
                "CAPTCHA",
                "captcha",
                "are you human",
                "Please enable JavaScript",
                "JavaScriptを有効",
                "アクセスが集中",
                "しばらくしてから",
            ]
            if any(w in t for w in blocked_words):
                raise RuntimeError("Fetch blocked or JS-required page returned (status=200).")

            return html
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff_sec * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff_sec * (attempt + 1))
                continue
            raise

    # ここには通常来ない
    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to fetch HTML.")


def looks_like_invalid_result_page(html: str) -> bool:
    """
    404以外で返る「該当レースなし」「未確定」「準備中」などを雑に検出する。
    """

    try:
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    except Exception:
        text = str(html or "")
    t = unicodedata.normalize("NFKC", text)
    keywords = [
        "該当するレースはありません",
        "該当するレースがありません",
        "ページが見つかりません",
        "ページが存在しません",
        "レース結果はありません",
        "レース結果がありません",
        "ただいま準備中",
        "準備中",
    ]
    return any(k in t for k in keywords)


def result_bundle_has_any_payout(bundle: Any, html: str) -> bool:
    """
    「レースが行われ、払戻が載っている」ページかどうかを判定する。
    - parse_payouts_bundle_from_result_html() の payouts/rows が空なら未実施/未確定/構造差分の可能性が高い
    """

    if isinstance(bundle, dict):
        rows = bundle.get("rows", [])
        if isinstance(rows, list) and len(rows) > 0:
            return True
        payouts = bundle.get("payouts", {})
        if isinstance(payouts, dict):
            for m in payouts.values():
                if isinstance(m, dict) and len(m) > 0:
                    return True

    # フォールバック: 「払戻」周辺に "xxx円" があるか
    try:
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    except Exception:
        text = str(html or "")
    t = unicodedata.normalize("NFKC", text)
    if "払戻" in t and _PAYOUT_YEN_RE.search(t):
        return True
    return False


def parse_race_context_from_html(url: str, html: str) -> Dict[str, Any]:
    """
    odds/result のURLとHTMLから「日付・競馬場・何R」をできる範囲で推定します。
    取得できない場合は空文字/None になります。
    """
    race_id = ""
    try:
        qs = parse_qs(urlparse(url).query)
        race_id = str(qs.get("race_id", [""])[0] or "")
    except Exception:
        race_id = ""

    soup = BeautifulSoup(html, "html.parser")
    candidates: List[str] = []
    if soup.title and soup.title.string:
        candidates.append(str(soup.title.string))
    for cls in ("RaceData01", "RaceData02", "RaceName", "RaceNum", "RacePlace", "RaceHead"):
        for tag in soup.select(f".{cls}"):
            txt = " ".join(tag.get_text(" ", strip=True).split())
            if txt:
                candidates.append(txt)
    page_text = " ".join(candidates)

    # 日付
    date_str = ""
    m = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", page_text)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        date_str = f"{y:04d}-{mo:02d}-{d:02d}"
    else:
        m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", page_text)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            date_str = f"{y:04d}-{mo:02d}-{d:02d}"
    if not date_str:
        # netkeiba(NAR) の race_id は YYYY + 場所コード2桁 + MM + DD + R2桁（例: 202655012404）
        m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})", race_id)
        if m:
            y, mo, d = int(m.group(1)), int(m.group(3)), int(m.group(4))
            if 1 <= mo <= 12 and 1 <= d <= 31:
                date_str = f"{y:04d}-{mo:02d}-{d:02d}"

    # 競馬場（候補リストから拾う）
    venues = [
        # NAR
        "大井",
        "川崎",
        "船橋",
        "浦和",
        "門別",
        "盛岡",
        "水沢",
        "金沢",
        "笠松",
        "名古屋",
        "園田",
        "姫路",
        "高知",
        "佐賀",
        # JRA
        "東京",
        "中山",
        "京都",
        "阪神",
        "札幌",
        "函館",
        "福島",
        "新潟",
        "中京",
        "小倉",
    ]
    venue = ""
    # まずは主要要素（title / 主要class）から拾う
    for v in venues:
        if v in page_text:
            venue = v
            break

    # 次に、ページ全体テキストから拾う（要素構造が変わっても壊れにくくする）
    if not venue:
        try:
            full_text = " ".join(soup.get_text(" ", strip=True).split())
        except Exception:
            full_text = ""
        for v in venues:
            if v in full_text:
                venue = v
                break

    # それでも取れない場合は race_id から推定（JRAの場コード）
    if not venue:
        m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})", race_id)
        if m:
            place_code = int(m.group(2))
            venue = JRA_PLACE_BY_CODE.get(place_code, "") or f"不明(場コード{place_code:02d})"

    # 何R
    race_no: Optional[int] = None
    m = re.search(r"(\d{1,2})\s*R", page_text)
    if m:
        race_no = int(m.group(1))

    return {"race_id": race_id, "date": date_str, "venue": venue, "race_no": race_no}


def parse_place_code_from_race_id(race_id: str) -> Optional[int]:
    """
    race_id から「場所コード（2桁）」を取り出します。
    例: 202606010301 -> 06
    """

    m = re.fullmatch(r"\d{4}(\d{2})\d{2}\d{2}\d{2}", str(race_id or ""))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


JRA_PLACE_BY_CODE: Dict[int, str] = {
    1: "札幌",
    2: "函館",
    3: "福島",
    4: "新潟",
    5: "東京",
    6: "中山",
    7: "中京",
    8: "京都",
    9: "阪神",
    10: "小倉",
}

NAR_PLACE_BY_CODE: Dict[int, str] = {
    30: "門別",
    35: "盛岡",
    36: "水沢",
    42: "浦和",
    43: "船橋",
    44: "大井",
    45: "川崎",
    46: "金沢",
    47: "笠松",
    48: "名古屋",
    50: "園田",
    51: "姫路",
    54: "高知",
    55: "佐賀",
}


def place_name_from_code(place_code: Optional[int]) -> str:
    if place_code is None:
        return ""
    try:
        return JRA_PLACE_BY_CODE.get(int(place_code), "")
    except Exception:
        return ""


def place_name_from_code_any(place_code: Optional[int]) -> str:
    if place_code is None:
        return ""
    try:
        code = int(place_code)
    except Exception:
        return ""
    return JRA_PLACE_BY_CODE.get(code, "") or NAR_PLACE_BY_CODE.get(code, "")


def parse_runner_count_from_result_html(html: str) -> Optional[int]:
    """
    結果ページHTMLから出走頭数（例: 16頭）を拾う。
    """
    if not html:
        return None
    try:
        soup = BeautifulSoup(html, "html.parser")
        texts: List[str] = []
        for cls in ("RaceData01", "RaceData02", "RaceData03", "RaceHead", "RaceName"):
            for tag in soup.select(f".{cls}"):
                t = tag.get_text(" ", strip=True)
                if t:
                    texts.append(t)
        if not texts:
            texts.append(soup.get_text(" ", strip=True))
        normalized = unicodedata.normalize("NFKC", " ".join(texts))
        candidates: List[int] = []
        for m in re.finditer(r"(\d{1,2})\s*頭", normalized):
            try:
                n = int(m.group(1))
            except Exception:
                continue
            if 1 <= n <= 30:
                candidates.append(n)
        if candidates:
            # 「1頭除外」などのノイズより、実際の出走頭数（通常は最大値）を優先
            return max(candidates)
    except Exception:
        return None
    return None


def count_valid_win_rows(win_rows: Any) -> Optional[int]:
    if not isinstance(win_rows, list):
        return None
    n = 0
    for r in win_rows:
        if not isinstance(r, dict):
            continue
        try:
            horse_no = int(r.get("馬番", 0))
            odds_val = float(r.get("単勝オッズ", 0.0))
        except Exception:
            continue
        if horse_no > 0 and odds_val > 0:
            n += 1
    return n if n > 0 else None


def parse_win_odds_rows_from_result_html(html: str) -> List[Dict[str, Any]]:
    """
    結果ページHTMLから「馬番/単勝オッズ/人気」を抽出する。
    単勝オッズページ取得に失敗した時のフォールバック用。
    """
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")

    def _norm(s: str) -> str:
        return unicodedata.normalize("NFKC", str(s or "")).strip()

    # 1) ヘッダ名（馬番/単勝オッズ）で列位置を特定して抽出
    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        if not trs:
            continue
        idx_horse = None
        idx_odds = None
        idx_pop = None
        header_row_idx = -1
        for i, tr in enumerate(trs):
            ths = tr.find_all("th")
            if len(ths) < 2:
                continue
            headers = [_norm(th.get_text(" ", strip=True)) for th in ths]
            for j, h in enumerate(headers):
                if idx_horse is None and "馬番" in h:
                    idx_horse = j
                if idx_odds is None and ("単勝オッズ" in h or ("単勝" in h and "オッズ" in h)):
                    idx_odds = j
                if idx_pop is None and "人気" in h:
                    idx_pop = j
            if idx_horse is not None and idx_odds is not None:
                header_row_idx = i
                break

        if idx_horse is None or idx_odds is None or header_row_idx < 0:
            continue

        rows: List[Dict[str, Any]] = []
        max_idx = max(idx_horse, idx_odds, idx_pop if idx_pop is not None else 0)
        for tr in trs[header_row_idx + 1 :]:
            tds = tr.find_all("td")
            if len(tds) <= max_idx:
                continue
            horse_no = parse_int_maybe(_norm(tds[idx_horse].get_text(" ", strip=True)))
            odds_val = parse_float_maybe(_norm(tds[idx_odds].get_text(" ", strip=True)))
            if horse_no is None or odds_val is None or odds_val <= 0:
                continue
            row: Dict[str, Any] = {"馬番": int(horse_no), "単勝オッズ": float(odds_val)}
            if idx_pop is not None and idx_pop < len(tds):
                pop = parse_int_maybe(_norm(tds[idx_pop].get_text(" ", strip=True)))
                if pop is not None and pop > 0:
                    row["人気"] = int(pop)
            rows.append(row)

        if rows:
            rows.sort(key=lambda r: (float(r["単勝オッズ"]), int(r["馬番"])))
            for i, r in enumerate(rows, start=1):
                if int(r.get("人気", 0)) <= 0:
                    r["人気"] = i
            return rows

    # 2) フォールバック: 行内の馬番 + 最終セルを単勝オッズ候補として拾う
    rows2: List[Dict[str, Any]] = []
    for tr in soup.find_all("tr"):
        horse_no: Optional[int] = None
        for sp in tr.select("span.UmaBan"):
            horse_no = parse_int_maybe(_norm(sp.get_text(" ", strip=True)))
            if horse_no is not None:
                break
        tds = tr.find_all("td")
        if horse_no is None and tds:
            horse_no = parse_int_maybe(_norm(tds[2].get_text(" ", strip=True) if len(tds) > 2 else ""))
        if horse_no is None or horse_no <= 0:
            continue
        odds_val: Optional[float] = None
        if tds:
            odds_val = parse_float_maybe(_norm(tds[-1].get_text(" ", strip=True)))
        if odds_val is None or odds_val <= 0:
            continue
        rows2.append({"馬番": int(horse_no), "単勝オッズ": float(odds_val)})

    rows2.sort(key=lambda r: (float(r["単勝オッズ"]), int(r["馬番"])))
    for i, r in enumerate(rows2, start=1):
        r["人気"] = i
    return rows2


def weekday_jp_from_ymd(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        y, m, d = [int(x) for x in date_str.split("-")]
        import datetime as _dt

        wd = _dt.date(y, m, d).weekday()  # Mon=0
        return ["月", "火", "水", "木", "金", "土", "日"][wd]
    except Exception:
        return ""


def date_md_jp_from_ymd(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        _, m, d = [int(x) for x in date_str.split("-")]
        return f"{m}月{d}日"
    except Exception:
        return ""


def date_for_sheets_from_ymd(date_str: str) -> str:
    """
    Google Sheets で「日付」として解釈されやすい表記にする（USER_ENTERED）。
    """
    if not date_str:
        return ""
    try:
        y, m, d = [int(x) for x in date_str.split("-")]
        return f"{y:04d}/{m:02d}/{d:02d}"
    except Exception:
        return str(date_str)


def _parse_numbers_from_bet_key(key: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", str(key))]


def filter_odds_json_to_top6(odds_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    単勝オッズ人気（win_oddsが小さい順、同値は馬番昇順）で上位6頭だけ残し、
    それ以外の馬は「最初から存在しない」扱いにする。

    - horses / tickets.win / tickets.place は上位6頭のみ
    - 組み合わせ系（quinella/wide/exacta/trio/trifecta）は、上位6頭だけで構成されたキーのみ残す
    """
    horses = odds_json.get("horses")
    if not isinstance(horses, dict):
        return odds_json

    entries: List[Tuple[int, float]] = []
    for k, v in horses.items():
        try:
            num = int(k)
            win_odds = float(v.get("win_odds", 0.0)) if isinstance(v, dict) else 0.0
        except Exception:
            continue
        if win_odds > 0:
            entries.append((num, win_odds))

    entries.sort(key=lambda t: (t[1], t[0]))
    allowed = {num for num, _ in entries[:6]}
    if not allowed:
        return odds_json

    out: Dict[str, Any] = json.loads(json.dumps(odds_json, ensure_ascii=False))

    # horses
    out_horses = out.get("horses")
    if isinstance(out_horses, dict):
        out["horses"] = {k: v for k, v in out_horses.items() if int(k) in allowed}

    tickets = out.get("tickets")
    if not isinstance(tickets, dict):
        return out

    # win/place (per horse)
    if isinstance(tickets.get("win"), dict):
        tickets["win"] = {k: v for k, v in tickets["win"].items() if int(k) in allowed}
    if isinstance(tickets.get("place"), dict):
        tickets["place"] = {k: v for k, v in tickets["place"].items() if int(k) in allowed}

    # combo markets
    def filter_map_by_nums(d: Any, need: int) -> Any:
        if not isinstance(d, dict):
            return d
        kept: Dict[str, Any] = {}
        for k, v in d.items():
            nums = _parse_numbers_from_bet_key(k)
            if len(nums) >= need and all(int(n) in allowed for n in nums[:need]):
                kept[k] = v
        return kept

    for key, need in (
        ("exacta", 2),
        ("quinella", 2),
        ("wide", 2),
        ("trio", 3),
        ("trifecta", 3),
    ):
        if key in tickets:
            tickets[key] = filter_map_by_nums(tickets.get(key), need)

    out["tickets"] = tickets
    return out


def pick_up_to_two_by_odds(candidates: List[Dict[str, Any]], odds_key: str) -> List[Dict[str, Any]]:
    """
    オッズで並べて「最大2個」選ぶ。
      - 1つ目: 最低オッズ（最も低い）
      - 2つ目: 最低〜最高の中間（= 上側中央値）
        - 個数が偶数なら高め（上側中央値）を採用
    """
    rows = []
    for c in candidates:
        try:
            odds_val = float(c.get(odds_key, 0.0))
        except Exception:
            continue
        if odds_val <= 0:
            continue
        rows.append({**c, odds_key: odds_val})
    if not rows:
        return []
    rows.sort(key=lambda r: float(r[odds_key]))
    if len(rows) == 1:
        return [rows[0]]
    mid_idx = len(rows) // 2  # 上側中央値（偶数なら高め）
    picked = [rows[0]]
    if mid_idx != 0:
        picked.append(rows[mid_idx])
    # 同じ行が重複した場合の保険
    out: List[Dict[str, Any]] = []
    seen = set()
    for r in picked:
        key = (str(r.get("種別", "")), str(r.get("買い目", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_service_account_info(
    *,
    uploaded: Any,
    local_path: str,
) -> Dict[str, Any]:
    """
    service account JSON を Streamlit uploader かローカルファイルから読み込む。
    """
    if uploaded is not None:
        return json.loads(uploaded.getvalue().decode("utf-8"))
    if local_path:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(f"サービスアカウントJSONが見つかりません: {local_path}")
        return json.loads(p.read_text(encoding="utf-8"))

    env_json = str(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "") or "").strip()
    if env_json:
        try:
            return json.loads(env_json)
        except Exception as e:
            raise ValueError(f"環境変数 GOOGLE_SERVICE_ACCOUNT_JSON のJSON解析に失敗しました: {e}") from e

    env_b64 = str(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64", "") or "").strip()
    if env_b64:
        try:
            decoded = base64.b64decode(env_b64).decode("utf-8")
            return json.loads(decoded)
        except Exception as e:
            raise ValueError(f"環境変数 GOOGLE_SERVICE_ACCOUNT_JSON_B64 の解析に失敗しました: {e}") from e

    raise ValueError("サービスアカウントJSONが未指定です（アップロード or ファイル名を指定）。")


def has_service_account_source(uploaded: Any, local_path: str) -> bool:
    if uploaded is not None:
        return True
    if bool(str(local_path or "").strip()):
        return True
    if bool(str(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "") or "").strip()):
        return True
    if bool(str(os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64", "") or "").strip()):
        return True
    return False


def install_google_sheets_deps() -> None:
    """
    Google Sheets 連携に必要な依存を pip でインストールする。
    Streamlit から呼ばれる想定（環境によっては失敗します）。
    """
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--user",
        "google-api-python-client",
        "google-auth",
    ]
    subprocess.check_call(cmd)


# ----------------------------
# Parse helpers
# ----------------------------
_RANGE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*$")
_FLOAT_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


def parse_range(s: str) -> Dict[str, float]:
    m = _RANGE_RE.match(s.strip())
    if not m:
        raise ValueError(f"range parse failed: {s}")
    return {"low": float(m.group(1)), "high": float(m.group(2))}


def parse_float_maybe(s: str) -> Optional[float]:
    """
    '返還' や '取消' など数値でない場合は None を返す。
    """
    txt = str(s).strip()
    if not txt:
        return None
    if any(x in txt for x in ("返還", "取消", "除外", "中止")):
        return None
    m = _FLOAT_RE.search(txt)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def parse_int_maybe(s: str) -> Optional[int]:
    txt = str(s).strip()
    if not txt:
        return None
    m = re.search(r"([0-9]+)", txt)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_range_maybe(s: str) -> Optional[Dict[str, float]]:
    txt = str(s).strip()
    if not txt:
        return None
    if any(x in txt for x in ("返還", "取消", "除外", "中止")):
        return None
    try:
        return parse_range(txt)
    except Exception:
        return None


def join_key(nums: List[str], sep: str) -> str:
    return sep.join(nums)


def parse_horses_table(table) -> Dict[str, Any]:
    horses: Dict[str, Any] = {}
    rows = table.find_all("tr")
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if len(tds) < 7:
            continue

        horse_no = tds[2].get_text(strip=True)
        name = tds[4].get_text(strip=True)

        win_odds = tds[5].get_text(strip=True)
        place_odds = tds[6].get_text(strip=True)

        if not horse_no or not win_odds:
            continue

        win_val = parse_float_maybe(win_odds)
        place_val = parse_range_maybe(place_odds)
        if win_val is None or place_val is None:
            # 返還/取消/除外などは「最初から存在しない」扱いにする
            continue

        horses[str(horse_no)] = {
            "name": name,  # 文字化けしててもOK（計算は馬番で回る）
            "win_odds": float(win_val),
            "place_odds": place_val,
        }
    return horses


def parse_combi_table(table) -> Tuple[str, Dict[str, Any]]:
    rows = table.find_all("tr")
    if len(rows) <= 1:
        return ("unknown", {})

    mapping: Dict[str, Any] = {}

    header_text = rows[0].get_text(" ", strip=True)
    has_wide_col = "ワイド" in header_text or "¥ï¥¤¥É" in header_text

    sample = rows[1]
    nums = [x.get_text(strip=True) for x in sample.select("span.UmaBan")]
    has_arrow = sample.select_one("span.Kaime_Arrow") is not None
    has_hyphen = sample.select_one("span.Hyphen") is not None

    kind = "unknown"
    if has_arrow and len(nums) == 3:
        kind = "trifecta"
    elif has_arrow and len(nums) == 2:
        kind = "exacta"
    elif has_hyphen and len(nums) == 3:
        kind = "trio"
    elif has_hyphen and len(nums) == 2 and has_wide_col:
        kind = "quinella_wide"
    elif has_hyphen and len(nums) == 2:
        kind = "quinella"

    for tr in rows[1:]:
        nums = [x.get_text(strip=True) for x in tr.select("span.UmaBan")]
        if not nums:
            continue

        odds_tds = tr.find_all("td", class_="Odds")

        if kind == "quinella_wide":
            if len(nums) != 2 or len(odds_tds) < 2:
                continue
            key = join_key(nums, "-")
            quinella_odds = parse_float_maybe(odds_tds[0].get_text(strip=True))
            wide_range = parse_range_maybe(odds_tds[1].get_text(strip=True))
            if quinella_odds is None or wide_range is None:
                continue
            mapping[key] = {"quinella": quinella_odds, "wide": wide_range}
        else:
            sep = ">" if kind in ("trifecta", "exacta") else "-"
            key = join_key(nums, sep)
            if not odds_tds:
                continue
            odds = parse_float_maybe(odds_tds[-1].get_text(strip=True))
            if odds is None:
                continue
            mapping[key] = odds

    return (kind, mapping)


def parse_wakuren_odds_from_html(html: str) -> Dict[str, float]:
    """
    枠連オッズページ（想定）から { "a-b": odds } を抽出する。
    ページ構造差分があるため、なるべく頑丈に「2つの枠番 + オッズ」を拾う。
    """

    soup = BeautifulSoup(html, "html.parser")
    out: Dict[str, float] = {}

    def _as_frame_digit(txt: str) -> Optional[int]:
        """
        文字列から「枠番(1..8)」をできるだけ頑丈に拾う。
        例: "1", "１", "1枠", "１枠", "枠1" など。
        """

        t = unicodedata.normalize("NFKC", str(txt or "").strip())
        if not t:
            return None

        # まず「数字のみ」を優先
        if t.isdigit():
            n = int(t)
            return n if 1 <= n <= 8 else None

        # 次に "1枠" / "枠1" などを許可
        m = re.search(r"\b([1-8])\b", t)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None

        # それでもダメなら、文字列中の最初の1桁数字を拾う（安全のため1..8限定）
        m = re.search(r"([1-8])", t)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    def _odds_from_node(node) -> Optional[float]:
        # class="Odds" を優先し、なければ小数を拾う
        try:
            s = node.get_text(" ", strip=True)
        except Exception:
            s = str(node or "")
        v = parse_float_maybe(s)
        if v is not None and v > 0:
            return float(v)
        # 小数点ありの数字を優先（人気の整数を避ける）
        m = re.findall(r"\b\d+\.\d+\b", unicodedata.normalize("NFKC", s))
        if m:
            try:
                v2 = float(m[-1])
                if v2 > 0:
                    return v2
            except Exception:
                return None
        return None

    # -------------------------
    # 1) リスト形式（行ごとに買い目がある）
    # -------------------------
    for tr in soup.find_all("tr"):
        # マトリクス表の行は Odds セルが複数あるため、先に除外する
        # （ここで拾うと "1 - 2.0 3.1" のような行から誤って "1-2" を作ってしまう）
        odds_tds = tr.find_all("td", class_="Odds")
        if len(odds_tds) >= 2:
            continue

        # odds（td.Oddsが無い構造もあるので、クラス名に Odds を含む要素も探す）
        odds_val: Optional[float] = None
        odds_nodes = tr.find_all(lambda tag: tag.name in ("td", "span") and tag.get("class") and any("Odds" in c for c in tag.get("class")))  # type: ignore[arg-type]
        if odds_nodes:
            odds_val = _odds_from_node(odds_nodes[-1])
        if odds_val is None:
            if odds_tds:
                odds_val = _odds_from_node(odds_tds[-1])
        if odds_val is None:
            # フォールバック: 行から「小数のオッズ」だけ拾う（人気などの整数を避ける）
            txt = unicodedata.normalize("NFKC", tr.get_text(" ", strip=True))
            ms = re.findall(r"\b\d+\.\d+\b", txt)
            if ms:
                try:
                    odds_val = float(ms[-1])
                except Exception:
                    odds_val = None
        if odds_val is None or odds_val <= 0:
            continue

        nums: List[int] = []
        # span側（クラス名が違っても拾えるように、1..8 だけを拾う）
        for sp in tr.select("span"):
            t = sp.get_text(strip=True)
            n = _as_frame_digit(t)
            if n is None:
                continue
            # "1人気" のような人気数字は除外（直後に人気が付く）
            parent_txt = unicodedata.normalize("NFKC", sp.parent.get_text(" ", strip=True) if sp.parent else t)
            if re.search(rf"{n}人気", parent_txt):
                continue
            nums.append(n)

        if len(nums) < 2:
            # フォールバック: 行テキストから「枠番ペア」を抽出（人気は除外）
            txt = unicodedata.normalize("NFKC", tr.get_text(" ", strip=True))
            txt = re.sub(r"\d人気", "", txt)
            # "1-2" 系
            m = re.search(r"\b([1-8])\s*[-‐‑–—ー]\s*([1-8])\b", txt)
            if m:
                nums = [int(m.group(1)), int(m.group(2))]
            else:
                # "1 2" 系
                m = re.search(r"\b([1-8])\s+([1-8])\b", txt)
                if m:
                    nums = [int(m.group(1)), int(m.group(2))]

        if len(nums) < 2:
            continue

        a, b = sorted(nums[:2])
        # 枠連は同枠（例: 5-5）も存在するため a==b を許可する
        if not (1 <= a <= 8 and 1 <= b <= 8):
            continue
        out[f"{a}-{b}"] = float(odds_val)

    if out:
        return out

    # -------------------------
    # 2) マトリクス形式（縦横に枠番、セルにオッズ）
    # -------------------------
    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        if len(trs) < 2:
            continue

        # ヘッダー行から列枠番を推定（先頭セルが空で、以降に 1..8 が並ぶ想定）
        header_cells = trs[0].find_all(["th", "td"])
        col_frames: List[int] = []
        for c in header_cells[1:]:
            n = _as_frame_digit(c.get_text(strip=True))
            if n is None:
                col_frames = []
                break
            col_frames.append(n)
        if len(col_frames) < 2:
            continue

        # 行ごとに先頭が行枠番、以降セルが対応オッズ
        for tr in trs[1:]:
            cells = tr.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            row_frame = _as_frame_digit(cells[0].get_text(strip=True))
            if row_frame is None:
                continue
            for j, cell in enumerate(cells[1 : 1 + len(col_frames)]):
                col_frame = col_frames[j]
                odds_val = _odds_from_node(cell)
                if odds_val is None or odds_val <= 0:
                    continue
                a, b = sorted([row_frame, col_frame])
                out[f"{a}-{b}"] = float(odds_val)

        if out:
            return out

    # -------------------------
    # 3) 正規表現フォールバック（埋め込みJSON/Next.jsなど）
    # -------------------------
    # 例: "kumiban":"1-2","odds":"4.2" / {"1-2":4.2} のような形を拾う
    text = unicodedata.normalize("NFKC", html)
    # kumiban -> odds
    for m in re.finditer(
        r'"kumiban"\s*:\s*"([1-8])\s*[-‐‑–—ー]\s*([1-8])".{0,80}?"odds"\s*:\s*"?(?P<odds>\d+(?:\.\d+)?)"?',
        text,
        flags=re.DOTALL,
    ):
        a = int(m.group(1))
        b = int(m.group(2))
        o = parse_float_maybe(m.group("odds"))
        if o is None or o <= 0:
            continue
        x, y = sorted([a, b])
        out[f"{x}-{y}"] = float(o)
    if out:
        return out

    # odds -> kumiban（順序逆）
    for m in re.finditer(
        r'"odds"\s*:\s*"?(?P<odds>\d+(?:\.\d+)?)"?.{0,120}?"kumiban"\s*:\s*"([1-8])\s*[-‐‑–—ー]\s*([1-8])"',
        text,
        flags=re.DOTALL,
    ):
        a = int(m.group(2))
        b = int(m.group(3))
        o = parse_float_maybe(m.group("odds"))
        if o is None or o <= 0:
            continue
        x, y = sorted([a, b])
        if x != y:
            out[f"{x}-{y}"] = float(o)
    if out:
        return out

    # "1-2": 4.2 形式
    for m in re.finditer(
        r'"([1-8])\s*[-‐‑–—ー]\s*([1-8])"\s*:\s*"?(?P<odds>\d+(?:\.\d+)?)"?',
        text,
    ):
        a = int(m.group(1))
        b = int(m.group(2))
        o = parse_float_maybe(m.group("odds"))
        if o is None or o <= 0:
            continue
        x, y = sorted([a, b])
        if x != y:
            out[f"{x}-{y}"] = float(o)
    if out:
        return out

    return out


def parse_wakuren_popular_rows_from_html(html: str) -> List[Dict[str, Any]]:
    """
    枠連（人気順表示）のHTMLから「人気/買い目/オッズ」を行単位で抽出する。
    - NAR の housiki=c99 などは「人気」列があり、そこから1..n人気を直接取れる。
    - 取れない場合は空リスト。
    """

    soup = BeautifulSoup(html, "html.parser")
    rows_out: List[Dict[str, Any]] = []

    def _pick_odds(nodes) -> Optional[float]:
        # "Odds" を含むノードが複数あるページがあるため、確率/人気/割合っぽい値を除外して拾う
        for node in nodes:
            try:
                txt = unicodedata.normalize("NFKC", node.get_text(" ", strip=True))
            except Exception:
                txt = ""
            if not txt:
                continue
            if any(x in txt for x in ("人気", "%", "投票", "票", "返還", "取消", "除外", "中止")):
                continue
            v = parse_float_maybe(txt)
            if v is not None and v > 0:
                return float(v)
        # 最後の手段: 末尾を拾う（ただし正の数のみ）
        if nodes:
            v2 = parse_float_maybe(nodes[-1].get_text(" ", strip=True))
            if v2 is not None and v2 > 0:
                return float(v2)
        return None

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        # popularity (人気)
        ninki: Optional[int] = None
        # まず class="Ninki" を優先（NAR の人気列）
        ninki_td = tr.find(
            "td",
            class_=lambda cls: cls
            and any("Ninki" in c for c in (cls if isinstance(cls, list) else [cls])),
        )
        if ninki_td is not None:
            ninki = parse_int_maybe(ninki_td.get_text(" ", strip=True))
        if ninki is None:
            # フォールバック: 「数字のみ」または「◯人気」を含むセルを探す
            for td in tds[:3]:
                txt = unicodedata.normalize("NFKC", td.get_text(" ", strip=True))
                if not txt:
                    continue
                if "人気" in txt or re.fullmatch(r"\d+", txt) is not None:
                    ninki = parse_int_maybe(txt)
                    break
        if ninki is None or ninki <= 0:
            continue

        # odds
        odds_val: Optional[float] = None
        odds_nodes = tr.select("td.Odds, span.Odds")
        if odds_nodes:
            odds_val = _pick_odds(odds_nodes)
        if odds_val is None:
            odds_nodes2 = tr.find_all(
                lambda tag: tag.name in ("td", "span")
                and tag.get("class")
                and any("Odds" in c for c in tag.get("class")),  # type: ignore[arg-type]
            )
            if odds_nodes2:
                odds_val = _pick_odds(odds_nodes2)
        if odds_val is None or odds_val <= 0:
            continue

        # combi (枠番2つ)
        nums: List[int] = []
        for sp in tr.select("span.UmaBan"):
            n = None
            try:
                n = int(unicodedata.normalize("NFKC", sp.get_text(strip=True)))
            except Exception:
                n = None
            if n is None or not (1 <= n <= 8):
                continue
            nums.append(n)

        if len(nums) < 2:
            # フォールバック（行テキストから抽出）
            txt = unicodedata.normalize("NFKC", tr.get_text(" ", strip=True))
            txt = re.sub(r"\d+人気", "", txt)
            m = re.search(r"\b([1-8])\s*[-‐‑–—ー]\s*([1-8])\b", txt)
            if m:
                nums = [int(m.group(1)), int(m.group(2))]
            else:
                m = re.search(r"\b([1-8])\s+([1-8])\b", txt)
                if m:
                    nums = [int(m.group(1)), int(m.group(2))]

        if len(nums) < 2:
            continue

        a, b = sorted(nums[:2])
        if not (1 <= a <= 8 and 1 <= b <= 8):
            continue

        rows_out.append({"人気": int(ninki), "買い目": f"{a}-{b}", "オッズ": float(odds_val)})

    # 人気順に整列・重複は先勝ち
    rows_out.sort(key=lambda r: int(r["人気"]))
    dedup: Dict[int, Dict[str, Any]] = {}
    for r in rows_out:
        k = int(r["人気"])
        if k not in dedup:
            dedup[k] = r
    return [dedup[k] for k in sorted(dedup.keys())]


def render_wakuren_ev_rows(rows: Sequence[Mapping[str, Any]], key_prefix: str) -> None:
    if not rows:
        st.warning("枠連EVを計算できませんでした（枠連オッズ抽出に失敗の可能性）。")
        return

    table_all = [
        {
            "race_id": str(r.get("race_id", "")),
            "人気": int(r.get("人気", 0)),
            "買い目": str(r.get("買い目", "")),
            "オッズ": float(r.get("オッズ", 0.0)),
            # 市場確率（正規化）: p = (1/odds) / Σ(1/odds)
            "市場確率(正規化)": float(r.get("市場確率", 0.0)),
            "市場確率(正規化%)": float(r.get("市場確率", 0.0)) * 100.0,
            # 念のため比較用（ユーザーが「逆数のまま」と感じた場合の確認用）
            "逆数(1/オッズ)": (1.0 / float(r.get("オッズ", 0.0))) if float(r.get("オッズ", 0.0)) > 0 else 0.0,
            "分母Σ(1/オッズ)推定": (
                (1.0 / float(r.get("オッズ", 0.0))) / float(r.get("市場確率", 0.0))
                if float(r.get("オッズ", 0.0)) > 0 and float(r.get("市場確率", 0.0)) > 0
                else 0.0
            ),
            "EV": float(r.get("EV", 0.0)),
        }
        for r in rows
    ]
    race_ids = sorted({t["race_id"] for t in table_all if t["race_id"]})
    if len(race_ids) >= 2:
        selected = st.selectbox("race_id を選択", race_ids, key=f"{key_prefix}_race_id")
        table = [t for t in table_all if t["race_id"] == selected]
        render_copy_button("枠連EVをコピー(TSV)", rows_to_tsv(table), key=f"{key_prefix}_copy")
        st.dataframe(table, use_container_width=True)
        with st.expander("全レースまとめ（TSV/表）", expanded=False):
            render_copy_button(
                "全レース枠連EVをコピー(TSV)", rows_to_tsv(table_all), key=f"{key_prefix}_all_copy"
            )
            st.dataframe(table_all, use_container_width=True)
    else:
        render_copy_button("枠連EVをコピー(TSV)", rows_to_tsv(table_all), key=f"{key_prefix}_copy")
        st.dataframe(table_all, use_container_width=True)


def compute_market_probs_and_ev_for_wakuren(
    odds_map: Mapping[str, float],
    top_n: int = 9,
) -> List[Dict[str, Any]]:
    """
    枠連のオッズから市場確率（逆数正規化）と EV = p*odds - 1 を計算。
    表示は人気（オッズ昇順）上位 top_n。

    ※ここでの「市場確率」は、枠連の「1番人気〜top_n人気（=オッズ昇順上位）」の範囲だけで正規化します。
      p_i = (1/odds_i) / Σ_{人気1..top_n}(1/odds)
    """

    rows: List[Dict[str, Any]] = []
    for key, odds in odds_map.items():
        try:
            o = float(odds)
        except Exception:
            continue
        if o <= 0:
            continue
        rows.append({"買い目": str(key), "odds": o})

    rows.sort(key=lambda r: float(r["odds"]))
    if not rows:
        return []

    # 表示は人気（オッズ昇順）上位のみ
    top_rows = rows[: int(top_n)] if int(top_n) > 0 else rows
    if not top_rows:
        return []

    qs = [1.0 / float(r["odds"]) for r in top_rows]
    total_q = sum(qs)  # 分母は人気1..top_nのみ
    if total_q <= 0:
        return []

    out: List[Dict[str, Any]] = []
    for idx, (r, q) in enumerate(zip(top_rows, qs), start=1):
        p = float(q) / float(total_q)
        ev = p * float(r["odds"]) - 1.0
        out.append(
            {
                "人気": idx,
                "買い目": r["買い目"],
                "オッズ": float(r["odds"]),
                "市場確率": p,
                "EV": ev,
            }
        )

    return out


def compute_market_probs_and_ev_for_wakuren_rows(
    popular_rows: Sequence[Mapping[str, Any]],
    top_n: int = 9,
) -> List[Dict[str, Any]]:
    """
    既に「人気順」に並んだ枠連行（人気/買い目/オッズ）から、市場確率とEVを計算する。
    分母（正規化）は「人気1..top_n」の範囲のみで計算する（ユーザー要望）。
    """

    rows = list(popular_rows)
    rows.sort(
        key=lambda r: int(r.get("人気", 0))
        if parse_int_maybe(r.get("人気", "")) is not None
        else 10**9
    )
    cleaned: List[Dict[str, Any]] = []
    for r in rows:
        ninki = parse_int_maybe(r.get("人気", ""))
        if ninki is None or ninki <= 0:
            continue
        o = parse_float_maybe(r.get("オッズ", ""))
        if o is None or o <= 0:
            continue
        cleaned.append(
            {
                "人気": int(ninki),
                "買い目": str(r.get("買い目", "")),
                "odds": float(o),
            }
        )
    if not cleaned:
        return []

    # 念のため「人気」順に再整列して、表示は1..top_n に振り直す
    cleaned.sort(key=lambda r: int(r["人気"]))
    show = cleaned[: int(top_n)] if int(top_n) > 0 else cleaned
    if not show:
        return []
    qs_show = [1.0 / float(r["odds"]) for r in show]
    total_q = sum(qs_show)  # 分母は人気1..top_nのみ
    if total_q <= 0:
        return []

    out: List[Dict[str, Any]] = []
    for r, q in zip(show, qs_show):
        p = float(q) / float(total_q)
        ev = p * float(r["odds"]) - 1.0
        out.append(
            {
                "人気": int(r["人気"]),
                "買い目": r["買い目"],
                "オッズ": float(r["odds"]),
                "市場確率": p,
                "EV": ev,
            }
        )
    return out


def compute_market_probs_and_ev_for_wakuren_popular_rows_with_denominator(
    popular_rows: Sequence[Mapping[str, Any]],
    odds_map_all: Mapping[str, float],
    top_n: int = 9,
) -> List[Dict[str, Any]]:
    """
    互換のため残す（現在は「人気1..top_nだけで正規化」する仕様）。
    """
    # odds_map_all は参照しない（分母は人気1..top_nのみ）
    _ = odds_map_all
    return compute_market_probs_and_ev_for_wakuren_rows(popular_rows, top_n=top_n)


def nar_win_odds_url(race_id: str) -> str:
    return f"https://nar.netkeiba.com/odds/?race_id={race_id}&type=b0"


def nar_wakuren_odds_url(race_id: str) -> str:
    # 互換のため残す（候補URLは compute_wakuren_ev_top9_for_race で複数試す）
    return f"https://nar.netkeiba.com/odds/index.html?type=b3&race_id={race_id}&housiki=c99"


def jra_wakuren_odds_url(race_id: str) -> str:
    # 互換のため残す（候補URLは compute_wakuren_ev_top9_for_race で複数試す）
    return f"https://race.netkeiba.com/odds/index.html?type=b3&race_id={race_id}&housiki=c99"


def jra_win_odds_url(race_id: str) -> str:
    # ユーザー指定の単勝オッズURL構造（JRA側）
    return f"https://race.netkeiba.com/odds/index.html?type=b1&race_id={race_id}&rf=shutuba_submenu"


def fetch_json(url: str, *, params: Optional[Dict[str, Any]] = None, timeout: int = 15) -> Any:
    """
    JSONを取得して dict/list を返します。
    netkeiba 系は UA が弱いとブロックされることがあるため、fetch_html と同じくブラウザ寄せヘッダにします。
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json,text/javascript,*/*;q=0.1",
        "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    retries = 2
    backoff_sec = 0.8
    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)

            if r.status_code in (403, 429):
                raise RuntimeError(
                    f"Fetch blocked (status={r.status_code}). Do not retry repeatedly; wait and try later."
                )

            if r.status_code in (408, 500, 502, 503, 504) and attempt < retries:
                time.sleep(backoff_sec * (attempt + 1))
                continue

            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError) as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff_sec * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(backoff_sec * (attempt + 1))
                continue
            raise

    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to fetch JSON.")


def decode_jra_odds_api_data(data: Any) -> Any:
    """
    JRA の odds API は compress=1 のとき data が base64+zlib 圧縮された JSON 文字列になる。
    それを復号して dict を返す。
    """

    if data is None:
        return None
    if isinstance(data, (dict, list)):
        return data

    s = str(data).strip()
    if not s:
        return None

    # すでにJSON文字列ならそのまま
    if s.startswith("{") or s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            pass

    # base64 -> zlib inflate -> json
    try:
        raw = base64.b64decode(s)
    except Exception as e:
        raise RuntimeError(f"base64 decode failed: {e}") from e

    dec: bytes
    try:
        dec = zlib.decompress(raw)
    except Exception:
        # 念のため raw deflate も試す
        try:
            dec = zlib.decompress(raw, -zlib.MAX_WBITS)
        except Exception as e:
            raise RuntimeError(f"zlib decompress failed: {e}") from e

    try:
        return json.loads(dec.decode("utf-8", errors="replace"))
    except Exception as e:
        raise RuntimeError(f"decoded JSON parse failed: {e}") from e


def fetch_jra_wakuren_odds_map_via_api(race_id: str) -> Dict[str, float]:
    """
    JRA（race.netkeiba.com）の枠連オッズはJSで注入される場合があるため、
    公式の odds API（api_get_jra_odds）を叩いて {\"a-b\": odds} を作る。
    """

    api_url = "https://race.netkeiba.com/api/api_get_jra_odds.html"
    params: Dict[str, Any] = {
        "pid": "api_get_jra_odds",
        "input": "UTF-8",
        "output": "json",
        "race_id": str(race_id),
        "type": "3",  # 枠連（oddsType=3）
        "action": "init",
        "sort": "odds",
        "compress": "1",
    }

    resp = fetch_json(api_url, params=params)
    if not isinstance(resp, dict):
        raise RuntimeError("JRA odds api returned non-object response.")

    status = str(resp.get("status", "") or "")
    reason = str(resp.get("reason", "") or "")
    data = resp.get("data", "")
    if not data:
        raise RuntimeError(f"JRA odds api returned empty data (status={status}, reason={reason}).")

    body = decode_jra_odds_api_data(data)
    if not isinstance(body, dict):
        raise RuntimeError("JRA odds api decode failed.")

    odds_root = body.get("odds", {})
    if not isinstance(odds_root, dict):
        raise RuntimeError("JRA odds api payload missing 'odds'.")

    odds_map_raw = odds_root.get("3", {})
    if not isinstance(odds_map_raw, dict):
        raise RuntimeError("JRA odds api payload missing odds['3'].")

    out: Dict[str, float] = {}
    for k, v in odds_map_raw.items():
        key = str(k)
        if len(key) < 4 or not key[:4].isdigit():
            continue
        try:
            a = int(key[:2])
            b = int(key[2:4])
        except Exception:
            continue
        if not (1 <= a <= 8 and 1 <= b <= 8):
            continue

        odds_val: Optional[float] = None
        if isinstance(v, (list, tuple)) and v:
            odds_val = parse_float_maybe(v[0])
        else:
            odds_val = parse_float_maybe(v)
        if odds_val is None or odds_val <= 0:
            continue

        x, y = sorted([a, b])
        out[f"{x}-{y}"] = float(odds_val)

    return out


def fetch_jra_wakuren_popular_rows_via_api(race_id: str) -> List[Dict[str, Any]]:
    """
    JRA odds API（api_get_jra_odds type=3）から「人気/買い目/オッズ」を取得する。
    """

    api_url = "https://race.netkeiba.com/api/api_get_jra_odds.html"
    params: Dict[str, Any] = {
        "pid": "api_get_jra_odds",
        "input": "UTF-8",
        "output": "json",
        "race_id": str(race_id),
        "type": "3",
        "action": "init",
        "sort": "odds",
        "compress": "1",
    }

    resp = fetch_json(api_url, params=params)
    if not isinstance(resp, dict):
        raise RuntimeError("JRA odds api returned non-object response.")
    data = resp.get("data", "")
    if not data:
        raise RuntimeError(
            f"JRA odds api returned empty data (status={resp.get('status')}, reason={resp.get('reason')})."
        )

    body = decode_jra_odds_api_data(data)
    if not isinstance(body, dict):
        raise RuntimeError("JRA odds api decode failed.")
    odds_root = body.get("odds", {})
    if not isinstance(odds_root, dict):
        raise RuntimeError("JRA odds api payload missing 'odds'.")
    odds_map_raw = odds_root.get("3", {})
    if not isinstance(odds_map_raw, dict):
        raise RuntimeError("JRA odds api payload missing odds['3'].")

    rows: List[Dict[str, Any]] = []
    for k, v in odds_map_raw.items():
        key = str(k)
        if len(key) < 4 or not key[:4].isdigit():
            continue
        try:
            a = int(key[:2])
            b = int(key[2:4])
        except Exception:
            continue
        if not (1 <= a <= 8 and 1 <= b <= 8):
            continue

        odds_val: Optional[float] = None
        rank_val: Optional[int] = None
        if isinstance(v, (list, tuple)) and v:
            odds_val = parse_float_maybe(v[0])
            if len(v) >= 3 and isinstance(v[2], int):
                rank_val = int(v[2])
        else:
            odds_val = parse_float_maybe(v)

        if odds_val is None or odds_val <= 0:
            continue

        x, y = sorted([a, b])
        rows.append({"人気": int(rank_val or 0), "買い目": f"{x}-{y}", "オッズ": float(odds_val)})

    # rank が入っているものを優先
    rows_ranked = [r for r in rows if int(r.get("人気", 0)) > 0]
    if rows_ranked:
        rows_ranked.sort(key=lambda r: int(r["人気"]))
        return rows_ranked

    # rankが無い場合はオッズ順で人気を振る
    rows.sort(key=lambda r: float(r["オッズ"]))
    for i, r in enumerate(rows, start=1):
        r["人気"] = i
    return rows


def fetch_jra_win_rows_via_api(race_id: str) -> List[Dict[str, Any]]:
    """
    JRA odds API（api_get_jra_odds type=1）から単勝オッズ行（馬番/単勝オッズ/人気）を取得する。
    HTML側がJS注入で空になる場合のフォールバック用。
    """

    api_url = "https://race.netkeiba.com/api/api_get_jra_odds.html"
    params: Dict[str, Any] = {
        "pid": "api_get_jra_odds",
        "input": "UTF-8",
        "output": "json",
        "race_id": str(race_id),
        "type": "1",  # 単勝
        "action": "init",
        "sort": "odds",
        "compress": "1",
    }

    resp = fetch_json(api_url, params=params)
    if not isinstance(resp, dict):
        raise RuntimeError("JRA odds api returned non-object response.")

    data = resp.get("data", "")
    if not data:
        raise RuntimeError(
            f"JRA odds api returned empty data (status={resp.get('status')}, reason={resp.get('reason')})."
        )

    body = decode_jra_odds_api_data(data)
    if not isinstance(body, dict):
        raise RuntimeError("JRA odds api decode failed.")

    odds_root = body.get("odds", {})
    if not isinstance(odds_root, dict):
        raise RuntimeError("JRA odds api payload missing 'odds'.")

    odds_map_raw = odds_root.get("1", {})
    if not isinstance(odds_map_raw, dict):
        raise RuntimeError("JRA odds api payload missing odds['1'].")

    rows: List[Dict[str, Any]] = []
    for k, v in odds_map_raw.items():
        key = str(k).strip()
        horse_no: Optional[int] = None

        # よくあるキー形式: "01" / "1" / "01..." に対応
        m = re.match(r"^0*([0-9]{1,2})$", key)
        if m:
            try:
                horse_no = int(m.group(1))
            except Exception:
                horse_no = None
        if horse_no is None:
            m = re.match(r"^0*([0-9]{1,2})", key)
            if m:
                try:
                    horse_no = int(m.group(1))
                except Exception:
                    horse_no = None

        if horse_no is None or not (1 <= horse_no <= 30):
            continue

        odds_val: Optional[float] = None
        rank_val: Optional[int] = None
        if isinstance(v, (list, tuple)) and v:
            odds_val = parse_float_maybe(v[0])
            if len(v) >= 3:
                rank_val = parse_int_maybe(v[2])
        else:
            odds_val = parse_float_maybe(v)

        if odds_val is None or odds_val <= 0:
            continue

        rows.append(
            {
                "馬番": int(horse_no),
                "単勝オッズ": float(odds_val),
                "_api_rank": int(rank_val) if rank_val is not None else 0,
            }
        )

    if not rows:
        raise RuntimeError("JRA odds api returned no valid win rows.")

    ranked = [r for r in rows if int(r.get("_api_rank", 0)) > 0]
    if ranked:
        ranked.sort(key=lambda r: (int(r.get("_api_rank", 0)), int(r.get("馬番", 0))))
        base = ranked
    else:
        rows.sort(key=lambda r: (float(r.get("単勝オッズ", 0.0)), int(r.get("馬番", 0))))
        base = rows

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(base, start=1):
        out.append(
            {
                "馬番": int(r.get("馬番", 0)),
                "単勝オッズ": float(r.get("単勝オッズ", 0.0)),
                "人気": int(i),
            }
        )
    return out


def parse_win_odds_from_html_generic(html: str) -> Dict[int, float]:
    """
    単勝オッズページ（NAR/JRAいずれも）から {馬番: 単勝オッズ} をできるだけ拾う。
    """

    soup = BeautifulSoup(html, "html.parser")
    out: Dict[int, float] = {}

    for tr in soup.find_all("tr"):
        nums = []
        for sp in tr.select("span.UmaBan"):
            try:
                nums.append(int(sp.get_text(strip=True)))
            except Exception:
                continue
        if not nums:
            # fallback: 最初の数字を馬番候補にする
            txt = unicodedata.normalize("NFKC", tr.get_text(" ", strip=True))
            m = re.match(r"^\s*([0-9]{1,2})\b", txt)
            if not m:
                continue
            try:
                nums = [int(m.group(1))]
            except Exception:
                continue

        horse_no = int(nums[0])
        if horse_no <= 0 or horse_no > 30:
            continue

        odds_val: Optional[float] = None
        odds_tds = tr.find_all("td", class_="Odds")
        if odds_tds:
            odds_val = parse_float_maybe(odds_tds[-1].get_text(strip=True))
        if odds_val is None:
            # fallback: 行から小数を拾う（人気など整数を避ける）
            txt = unicodedata.normalize("NFKC", tr.get_text(" ", strip=True))
            ms = re.findall(r"\b\d+\.\d+\b", txt)
            if ms:
                try:
                    odds_val = float(ms[-1])
                except Exception:
                    odds_val = None

        if odds_val is None or odds_val <= 0:
            continue
        out[horse_no] = float(odds_val)

    return out


def compute_win_odds_rows_from_map(win_odds_by_no: Mapping[int, float]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for no, o in win_odds_by_no.items():
        try:
            n = int(no)
            v = float(o)
        except Exception:
            continue
        if n <= 0 or v <= 0:
            continue
        rows.append({"馬番": n, "単勝オッズ": v})
    rows.sort(key=lambda r: (float(r["単勝オッズ"]), int(r["馬番"])))
    for i, r in enumerate(rows, start=1):
        r["人気"] = i
    return rows


def compute_nar_win_odds_rows(odds_json: Mapping[str, Any]) -> List[Dict[str, Any]]:
    horses = odds_json.get("horses", {})
    rows: List[Dict[str, Any]] = []
    if isinstance(horses, dict):
        for k, v in horses.items():
            if not isinstance(v, dict):
                continue
            try:
                no = int(k)
                o = float(v.get("win_odds", 0.0))
            except Exception:
                continue
            if o <= 0:
                continue
            rows.append({"馬番": no, "単勝オッズ": o})
    rows.sort(key=lambda r: (float(r["単勝オッズ"]), int(r["馬番"])))
    for i, r in enumerate(rows, start=1):
        r["人気"] = i
    return rows


def fetch_win_rows_for_race(race_id: str) -> List[Dict[str, Any]]:
    """
    race_id から単勝オッズ行（馬番/単勝オッズ/人気）を取得する。
    """
    place_code = parse_place_code_from_race_id(race_id)
    likely_jra = place_code is not None and 1 <= int(place_code) <= 10

    def _try_jra_html() -> List[Dict[str, Any]]:
        win_url = jra_win_odds_url(race_id)
        win_html = fetch_html(win_url)
        win_map = parse_win_odds_from_html_generic(win_html)
        return compute_win_odds_rows_from_map(win_map)

    def _try_nar_html() -> List[Dict[str, Any]]:
        win_url = nar_win_odds_url(race_id)
        win_html = fetch_html(win_url)
        win_odds_json = build_odds_json(win_url, win_html)
        return compute_nar_win_odds_rows(win_odds_json)

    def _try_jra_api() -> List[Dict[str, Any]]:
        return fetch_jra_win_rows_via_api(race_id)

    attempts = ([_try_jra_html, _try_jra_api, _try_nar_html] if likely_jra else [_try_nar_html, _try_jra_html, _try_jra_api])
    for fn in attempts:
        try:
            rows = fn()
            if rows:
                return rows
        except Exception:
            continue
    return []


def compute_gini_coefficient(values: List[float]) -> Optional[float]:
    xs: List[float] = []
    for v in values:
        try:
            x = float(v)
        except Exception:
            continue
        if x > 0:
            xs.append(x)
    n = len(xs)
    if n <= 0:
        return None
    s = sum(xs)
    if s <= 0:
        return None
    xs.sort()
    weighted = 0.0
    for i, x in enumerate(xs, start=1):
        weighted += float(i) * float(x)
    g = (2.0 * weighted) / (float(n) * float(s)) - (float(n + 1) / float(n))
    if g < 0:
        g = 0.0
    if g > 1:
        g = 1.0
    return float(g)


def compute_race_gini_from_win_rows(
    race_id: str,
    win_rows: Any,
    runner_count_hint: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(win_rows, list):
        return None
    valid: List[Dict[str, Any]] = []
    for r in win_rows:
        if not isinstance(r, dict):
            continue
        try:
            no = int(r.get("馬番", 0))
            odds = float(r.get("単勝オッズ", 0.0))
        except Exception:
            continue
        if no > 0 and odds > 0:
            valid.append({"馬番": no, "単勝オッズ": odds})
    if not valid:
        return None
    valid.sort(key=lambda x: (float(x["単勝オッズ"]), int(x["馬番"])))
    observed_count = len(valid)
    runner_count = int(runner_count_hint) if runner_count_hint is not None and int(runner_count_hint) > 0 else observed_count
    # 指示どおり「出走頭数の半分」（奇数は切り捨て）
    top_count = max(1, runner_count // 2)
    # 頭数が少ないと1頭だけになりジニ係数が必ず0になるため、2頭以上いる場合は最低2頭で計算する
    if observed_count >= 2 and top_count < 2:
        top_count = 2
    if top_count > observed_count:
        top_count = observed_count
    top_half = valid[:top_count]

    # 先頭が同オッズで0になりやすいケースを避けるため、必要なら対象を後ろへ広げる
    distinct = {float(x["単勝オッズ"]) for x in top_half}
    if len(distinct) < 2 and observed_count > top_count:
        for i in range(top_count, observed_count):
            top_half.append(valid[i])
            distinct.add(float(valid[i]["単勝オッズ"]))
            top_count += 1
            if len(distinct) >= 2:
                break

    g = compute_gini_coefficient([float(x["単勝オッズ"]) for x in top_half])
    if g is None:
        return None
    return {
        "race_id": str(race_id),
        "出走頭数": int(runner_count),
        "ジニ対象頭数": int(top_count),
        "ジニ係数": float(g),
    }


def attach_market_prob_to_win_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    単勝オッズ行（{"馬番","単勝オッズ",...}）に、市場確率 p_i=(1/o_i)/Σ(1/o) を付与する。
    分母は rows に含まれる「全馬」で計算する。
    """
    cleaned: List[Tuple[int, float]] = []
    for r in rows:
        try:
            num = int(r.get("馬番", 0))
            o = float(r.get("単勝オッズ", 0.0))
        except Exception:
            continue
        if num <= 0 or o <= 0:
            continue
        cleaned.append((num, o))
    if not cleaned:
        return rows

    total_q = sum(1.0 / o for _, o in cleaned)
    if total_q <= 0:
        return rows

    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            o = float(r.get("単勝オッズ", 0.0))
        except Exception:
            o = 0.0
        p = 0.0
        if o > 0:
            p = (1.0 / o) / float(total_q)
        out.append(
            {
                **r,
                "市場確率": float(p),
                "市場確率(%)": float(p) * 100.0,
            }
        )
    return out


def compute_wakuren_ev_top9_for_race(race_id: str) -> Dict[str, Any]:
    """
    race_id から（NAR/JRA 両対応）
      - 枠連オッズ → 市場確率（逆数正規化） → EV（上位9人気）
      - （NARのみ）単勝オッズ（参考）
    をまとめて返す。
    """

    place_code = parse_place_code_from_race_id(race_id)
    likely_jra = place_code is not None and 1 <= int(place_code) <= 10

    win_rows: List[Dict[str, Any]] = []
    # 単勝:
    # - NAR: b0（build_odds_json が安定）
    # - JRA: b1（汎用HTMLパーサで拾える範囲で）
    if likely_jra:
        try:
            win_url = jra_win_odds_url(race_id)
            win_html = fetch_html(win_url)
            win_map = parse_win_odds_from_html_generic(win_html)
            win_rows = compute_win_odds_rows_from_map(win_map)
        except Exception:
            try:
                win_url = nar_win_odds_url(race_id)
                win_html = fetch_html(win_url)
                win_odds_json = build_odds_json(win_url, win_html)
                win_rows = compute_nar_win_odds_rows(win_odds_json)
            except Exception:
                win_rows = []
    else:
        try:
            win_url = nar_win_odds_url(race_id)
            win_html = fetch_html(win_url)
            win_odds_json = build_odds_json(win_url, win_html)
            win_rows = compute_nar_win_odds_rows(win_odds_json)
        except Exception:
            try:
                win_url = jra_win_odds_url(race_id)
                win_html = fetch_html(win_url)
                win_map = parse_win_odds_from_html_generic(win_html)
                win_rows = compute_win_odds_rows_from_map(win_map)
            except Exception:
                win_rows = []

    # 単勝の市場確率（p_i=(1/o_i)/Σ(1/o)）を付与
    win_rows = attach_market_prob_to_win_rows(win_rows)

    # 枠連（b3）
    # - ページURLは環境/導線で微妙に変わるため、複数候補を試して「実際に抽出できたもの」を採用する
    nar_wak_url_candidates = [
        f"https://nar.netkeiba.com/odds/index.html?type=b3&race_id={race_id}&housiki=c99",
        f"https://nar.netkeiba.com/odds/index.html?type=b3&race_id={race_id}",
        f"https://nar.netkeiba.com/odds/index.html?type=b2&race_id={race_id}&housiki=c99",
        f"https://nar.netkeiba.com/odds/index.html?type=b2&race_id={race_id}",
    ]
    jra_wak_url_candidates = [
        # type はサイト側の差分があり得るので b2/b3 の両方を試す（ただしAPIが取れればそちらを優先）
        f"https://race.netkeiba.com/odds/index.html?type=b3&race_id={race_id}&rf=shutuba_submenu",
        f"https://race.netkeiba.com/odds/index.html?type=b3&race_id={race_id}",
        f"https://race.netkeiba.com/odds/index.html?type=b2&race_id={race_id}&rf=shutuba_submenu",
        f"https://race.netkeiba.com/odds/index.html?type=b2&race_id={race_id}",
    ]
    wak_url_candidates = nar_wak_url_candidates + jra_wak_url_candidates

    wak_url: str = wak_url_candidates[0]
    wak_odds_map: Dict[str, float] = {}
    last_html: str = ""
    attempt_errors: List[str] = []
    api_err: str = ""

    # JRAはHTMLにオッズが無い（JS注入）ことが多いので、まずAPIを優先する
    if likely_jra:
        try:
            jra_popular_rows = fetch_jra_wakuren_popular_rows_via_api(race_id)
            if jra_popular_rows:
                wak_odds_map = {str(r["買い目"]): float(r["オッズ"]) for r in jra_popular_rows}
                wak_url = f"https://race.netkeiba.com/odds/index.html?type=b3&race_id={race_id}&rf=shutuba_submenu"
        except Exception as e:
            api_err = f"{type(e).__name__}: {e}"

    # HTMLスクレイピング（APIがダメ/未対応の場合のフォールバック）
    html_candidates = (jra_wak_url_candidates + nar_wak_url_candidates) if likely_jra else (nar_wak_url_candidates + jra_wak_url_candidates)
    if not wak_odds_map:
        for cand in html_candidates:
            try:
                wak_html = fetch_html(cand)
                last_html = wak_html
                # まず「全買い目」を取る（市場確率の分母に必要）
                full_map = parse_wakuren_odds_from_html(wak_html)

                # 次に「人気順行」を取る（表示上の人気1..9の整合性が上がる）
                popular_rows = parse_wakuren_popular_rows_from_html(wak_html)

                if full_map:
                    wak_odds_map = full_map
                    wak_url = cand
                    break

                # full_map が取れないページでも、最低限の表示のため人気順行だけ採用
                if popular_rows:
                    wak_odds_map = {str(r["買い目"]): float(r["オッズ"]) for r in popular_rows}
                    wak_url = cand
                    break
            except Exception as e:
                attempt_errors.append(f"{cand} -> {type(e).__name__}: {e}")
                continue

    # まだ取れていなければ、念のためAPIも試す（race_id 判定が外れている場合に備える）
    if not wak_odds_map and not likely_jra:
        try:
            jra_popular_rows = fetch_jra_wakuren_popular_rows_via_api(race_id)
            if jra_popular_rows:
                wak_odds_map = {str(r["買い目"]): float(r["オッズ"]) for r in jra_popular_rows}
                wak_url = f"https://race.netkeiba.com/odds/index.html?type=b3&race_id={race_id}&rf=shutuba_submenu"
        except Exception as e:
            if not api_err:
                api_err = f"{type(e).__name__}: {e}"

    if not wak_odds_map:
        # 最終的に抽出できなかった場合は、デバッグできるようにURLを含めて例外にする
        msg = "枠連オッズ抽出に失敗しました。"
        msg += " URL候補=" + ", ".join(wak_url_candidates)
        if attempt_errors:
            msg += " / errors=" + " | ".join(attempt_errors[:5])
            if len(attempt_errors) > 5:
                msg += f" ...(+{len(attempt_errors) - 5})"
        if api_err:
            msg += f" / api_error={api_err}"
        if last_html:
            msg += "（取得はできましたが、オッズが見つかりませんでした）"
        raise RuntimeError(msg)

    # 人気順行がHTML/APIで取れるケースでは、その順序を尊重してEV計算する
    wak_ev_rows: List[Dict[str, Any]] = []
    if last_html:
        pop_rows = parse_wakuren_popular_rows_from_html(last_html)
        if pop_rows:
            wak_ev_rows = compute_market_probs_and_ev_for_wakuren_rows(pop_rows, top_n=9)
    if not wak_ev_rows and likely_jra:
        try:
            pop_rows = fetch_jra_wakuren_popular_rows_via_api(race_id)
            if pop_rows:
                wak_ev_rows = compute_market_probs_and_ev_for_wakuren_rows(pop_rows, top_n=9)
        except Exception:
            wak_ev_rows = []
    if not wak_ev_rows:
        wak_ev_rows = compute_market_probs_and_ev_for_wakuren(wak_odds_map, top_n=9)

    # 表示しやすい形に変換
    wak_ev_rows2: List[Dict[str, Any]] = []
    for r in wak_ev_rows:
        wak_ev_rows2.append(
            {
                "race_id": race_id,
                "wakuren_url": wak_url,
                "人気": int(r["人気"]),
                "買い目": str(r["買い目"]),
                "オッズ": float(r["オッズ"]),
                "市場確率": float(r["市場確率"]),
                "EV": float(r["EV"]),
            }
        )

    return {"win_rows": win_rows, "wakuren_ev_rows": wak_ev_rows2}


def build_odds_json(url: str, html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    tables = soup.find_all("table", class_="RaceOdds_HorseList_Table")
    if not tables:
        raise RuntimeError("No tables found. Page may be dynamically generated or blocked.")

    tickets: Dict[str, Any] = {
        "win": {},
        "place": {},
        "exacta": {},
        "quinella": {},
        "wide": {},
        "trifecta": {},
        "trio": {},
    }

    horses_table = None
    for t in tables:
        if t.select_one("th.Horse_Name") is not None:
            horses_table = t
            break
    if horses_table is None:
        raise RuntimeError("Horses table not found.")

    horses = parse_horses_table(horses_table)

    for no, info in horses.items():
        tickets["win"][no] = float(info["win_odds"])
        tickets["place"][no] = info["place_odds"]

    for t in tables:
        if t is horses_table:
            continue
        kind, mapping = parse_combi_table(t)
        if kind == "trifecta":
            tickets["trifecta"].update(mapping)
        elif kind == "exacta":
            tickets["exacta"].update(mapping)
        elif kind == "trio":
            tickets["trio"].update(mapping)
        elif kind == "quinella":
            tickets["quinella"].update(mapping)
        elif kind == "quinella_wide":
            for key, v in mapping.items():
                tickets["quinella"][key] = v["quinella"]
                tickets["wide"][key] = v["wide"]

    return {
        "meta": {
            "source_url": url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
        "horses": horses,
        "tickets": tickets,
    }


# ----------------------------
# Parse payouts (result page)
# ----------------------------
_PAYOUT_YEN_RE = re.compile(r"([0-9]{1,3}(?:,[0-9]{3})*)\s*円")


def _parse_yen(value: str) -> Optional[int]:
    m = _PAYOUT_YEN_RE.search(value)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def parse_user_bet_keys(kind: str, raw: str) -> List[str]:
    """
    UIの買い目入力（自由形式）を、払戻パース側と同じ「正規化キー」に揃えます。

    対応券種:
      - 単勝: "8" -> "8"
      - 枠連/馬連: "6-8", "6 8" -> "6-8"（順不同）
      - 馬単: "8>12", "8→12", "8 12" -> "8→12"（順序あり）
    """

    txt = unicodedata.normalize("NFKC", str(raw or ""))
    # よくある区切りを揃える
    txt = txt.replace(">", "→").replace("⇒", "→").replace("→", "→")
    txt = re.sub(r"[，、;/]+", ",", txt)
    txt = txt.replace("\n", ",")
    parts = [p.strip() for p in txt.split(",") if p.strip()]

    out: List[str] = []
    for p in parts:
        nums = [int(x) for x in re.findall(r"\d+", p)]
        key = _normalize_bet_key(kind, nums[:3])
        # 馬単は順序が重要なので、数字が2つあればその順序で扱う（例: "8 12"）
        if key is None and kind == "馬単" and len(nums) >= 2:
            key = f"{nums[0]}→{nums[1]}"
        if key is None:
            continue
        out.append(key)

    # 重複除去（入力順は維持）
    seen: set[str] = set()
    uniq: List[str] = []
    for k in out:
        if k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    return uniq


def parse_user_bet_ranks(kind: str, raw: str) -> List[List[int]]:
    """
    UIの買い目入力（人気順位ベース）をパースして、整数のリスト（組）にします。
    例:
      - 単勝: "1,2" -> [[1],[2]]
      - 枠連/馬連: "1-3, 2 4" -> [[1,3],[2,4]]
      - 馬単: "1>2, 3→1" -> [[1,2],[3,1]]  (順序あり)
    """

    txt = unicodedata.normalize("NFKC", str(raw or ""))
    txt = txt.replace(">", "→").replace("⇒", "→")
    txt = re.sub(r"[，、;/]+", ",", txt)
    txt = txt.replace("\n", ",")
    parts = [p.strip() for p in txt.split(",") if p.strip()]

    out: List[List[int]] = []
    for p in parts:
        nums = [int(x) for x in re.findall(r"\d+", p)]
        if kind == "単勝":
            if len(nums) >= 1:
                out.append([nums[0]])
        elif kind in ("枠連", "馬連"):
            if len(nums) >= 2:
                out.append([nums[0], nums[1]])
        elif kind == "馬単":
            if len(nums) >= 2:
                out.append([nums[0], nums[1]])

    # 重複除去（入力順維持）
    seen: set[tuple[int, ...]] = set()
    uniq: List[List[int]] = []
    for xs in out:
        t = tuple(xs)
        if t in seen:
            continue
        seen.add(t)
        uniq.append(xs)
    return uniq


def parse_user_ticket_popularity_ranks(raw: str) -> List[int]:
    """
    UIの買い目入力（馬券そのものの人気順位）をパースします。
    例: "1,2,5" -> [1,2,5]
    """

    txt = unicodedata.normalize("NFKC", str(raw or ""))
    txt = re.sub(r"[，、;/\\s]+", ",", txt)
    txt = txt.replace("\n", ",")
    nums = [int(x) for x in re.findall(r"\d+", txt)]
    out: List[int] = []
    seen: set[int] = set()
    for n in nums:
        if n <= 0:
            continue
        if n in seen:
            continue
        seen.add(n)
        out.append(int(n))
    return out


def compute_settlement_from_payout_rows(
    payout_rows: Sequence[Mapping[str, Any]],
    user_bets: Mapping[str, Sequence[int]],
    kinds: Sequence[str],
    stake_per_bet_yen: int = 100,
) -> Dict[str, Any]:
    """
    払戻行（bundle["rows"]）と「馬券の人気順位ベースの買い目」を比較して、
    掛け金・払戻・利益を集計します。

    前提:
      - user_bets[kind] は [1,3,...] のような「馬券そのものの人気順位」
      - payout_rows の row["人気"] は "1人気" のような文字列
      - payout_rows の row["払戻金"] は int または数値化可能
    """

    selected_by_kind: Dict[str, List[int]] = {}
    for kind in kinds:
        selected_by_kind[kind] = [int(x) for x in (user_bets.get(kind, []) or []) if int(x) > 0]

    stake_by_kind: Dict[str, int] = {
        kind: stake_per_bet_yen * len(selected_by_kind[kind]) for kind in kinds
    }
    stake_total = sum(stake_by_kind.values())

    returned_by_kind: Dict[str, int] = {kind: 0 for kind in kinds}
    hit_details: Dict[str, List[str]] = {kind: [] for kind in kinds}

    for row in payout_rows:
        kind = str(row.get("券種", "")).strip()
        if kind not in kinds:
            continue
        pop_rank = parse_int_maybe(str(row.get("人気", "") or ""))
        if pop_rank is None:
            continue
        if pop_rank not in set(selected_by_kind[kind]):
            continue
        amount = parse_int_maybe(str(row.get("払戻金", 0)))
        if amount is None:
            continue
        returned_by_kind[kind] += int(amount)
        bet_disp = str(row.get("買い目", "") or "")
        hit_details[kind].append(f"{pop_rank}人気:{bet_disp}={int(amount)}円")

    returned_total = sum(returned_by_kind.values())
    profit_total = int(returned_total) - int(stake_total)

    return {
        "kinds": list(kinds),
        "selected_by_kind": selected_by_kind,
        "stake_by_kind": stake_by_kind,
        "stake_total": int(stake_total),
        "returned_by_kind": returned_by_kind,
        "returned_total": int(returned_total),
        "profit_total": int(profit_total),
        "hit_details": hit_details,
    }


def build_popularity_from_odds_html(odds_url: str, odds_html: str) -> Dict[str, Any]:
    """
    オッズHTMLから「人気順位→馬番」を作る。
    popularity_horses: [馬番(1人気), 馬番(2人気), ...]
    """

    odds_json = build_odds_json(odds_url, odds_html)
    horses = odds_json.get("horses", {})
    entries: List[Tuple[int, float]] = []
    if isinstance(horses, dict):
        for k, v in horses.items():
            if not isinstance(v, dict):
                continue
            try:
                num = int(k)
                win_odds = float(v.get("win_odds", 0.0))
            except Exception:
                continue
            if win_odds > 0:
                entries.append((num, win_odds))
    entries.sort(key=lambda t: (t[1], t[0]))
    popularity_horses = [num for num, _ in entries]
    return {"odds_json": odds_json, "popularity_horses": popularity_horses}


def waku_from_horse_number(n_horses: int, horse_number: int) -> Optional[int]:
    """
    馬番 → 枠番（枠連用）の簡易変換。
    JRA/NAR共通で使える「先頭が単枠、後ろが複数頭」になる一般的な割当:
      - n<=8: 枠=馬番
      - 9<=n<=16: 先頭(16-n)枠が単枠、残りは2頭ずつ
      - n>=17: 1..16 を上と同様に割り当て、余り(17,18...)は枠8に入れる
    """

    try:
        n = int(n_horses)
        no = int(horse_number)
    except Exception:
        return None
    if n <= 0 or no <= 0 or no > n:
        return None
    if n <= 8:
        return no

    if n >= 17:
        if no >= 15:
            return 8
        # 1..14 は 16頭の割当と同様に扱う
        n = 16

    singles = 16 - n  # 0..7
    if no <= singles:
        return no
    # singles+1 以降は2頭ずつ
    idx = no - singles - 1  # 0-based within paired section
    frame = singles + 1 + (idx // 2)
    if frame > 8:
        frame = 8
    return frame


def ranks_to_bet_keys(
    kind: str,
    rank_groups: Sequence[Sequence[int]],
    popularity_horses: Sequence[int],
    n_horses: int,
) -> List[str]:
    """
    人気順位の組を、払戻照合用の「買い目キー」に変換する。
    """

    keys: List[str] = []
    for ranks in rank_groups:
        # 人気順位 -> 馬番
        horse_nums: List[int] = []
        ok = True
        for r in ranks:
            if r <= 0 or r > len(popularity_horses):
                ok = False
                break
            horse_nums.append(int(popularity_horses[r - 1]))
        if not ok:
            continue

        # 枠連だけは枠番に変換
        if kind == "枠連":
            frames: List[int] = []
            for hn in horse_nums:
                w = waku_from_horse_number(n_horses, hn)
                if w is None:
                    ok = False
                    break
                frames.append(int(w))
            if not ok:
                continue
            key = _normalize_bet_key("枠連", frames)
        else:
            key = _normalize_bet_key(kind, horse_nums)
        if key:
            keys.append(key)

    # 重複除去
    seen: set[str] = set()
    uniq: List[str] = []
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        uniq.append(k)
    return uniq


def _normalize_bet_key(kind: str, nums: List[int]) -> Optional[str]:
    if kind in ("単勝", "複勝"):
        if len(nums) != 1:
            return None
        return str(nums[0])
    if kind in ("枠連", "馬連", "ワイド"):
        if len(nums) != 2:
            return None
        a, b = sorted(nums)
        return f"{a}-{b}"
    if kind == "三連複":
        if len(nums) != 3:
            return None
        a, b, c = sorted(nums)
        return f"{a}-{b}-{c}"
    if kind == "馬単":
        if len(nums) != 2:
            return None
        return f"{nums[0]}→{nums[1]}"
    if kind == "三連単":
        if len(nums) != 3:
            return None
        return f"{nums[0]}→{nums[1]}→{nums[2]}"
    return None


def parse_payouts_from_result_html(html: str) -> Dict[str, Dict[str, int]]:
    """
    https://nar.netkeiba.com/race/result.html?... の払い戻し欄をパースして
    {券種: {買い目: 払戻(円/100円)}} を返します。
    """
    soup = BeautifulSoup(html, "html.parser")

    targets = {"単勝", "複勝", "馬連", "馬単", "ワイド", "三連複", "三連単"}
    payouts: Dict[str, Dict[str, int]] = {k: {} for k in targets}

    # テーブル構造はページにより少し変わるので、行ベースでスキャンする
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) < 3:
            continue

        kind = cells[0].get_text(strip=True)
        if kind not in targets:
            continue

        bet_lines = cells[1].get_text("\n", strip=True).splitlines()
        pay_lines = cells[2].get_text("\n", strip=True).splitlines()

        # 行数が合わない場合は、全文から拾えるだけ拾って対応する
        if len(bet_lines) != len(pay_lines):
            bet_lines = [cells[1].get_text(" ", strip=True)]
            pay_lines = [cells[2].get_text(" ", strip=True)]

        for bet_text, pay_text in zip(bet_lines, pay_lines):
            nums = [int(x) for x in re.findall(r"\d+", bet_text)]
            key = _normalize_bet_key(kind, nums)
            if not key:
                continue
            yen = _parse_yen(pay_text)
            if yen is None:
                continue
            payouts[kind][key] = yen

    return payouts


def parse_payouts_bundle_from_result_html(html: str) -> Dict[str, Any]:
    """
    払い戻し欄をパースして、表示用 rows と照合用 payouts をまとめて返します。
    rows は以下の形式:
      [{"券種": "...", "買い目": "...", "払戻金": 160, "人気": "1人気"}, ...]
    payouts は照合用の正規化キー:
      {"単勝": {"8": 160}, "馬単": {"8→12": 220}, ...}
    """
    soup = BeautifulSoup(html, "html.parser")

    # UI / 購入リスト側の券種表記に寄せる（照合が簡単になる）
    canonical_kinds = {"単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"}
    payouts: Dict[str, Dict[str, int]] = {k: {} for k in canonical_kinds}
    rows_out: List[Dict[str, Any]] = []

    kind_order = {
        "単勝": 1,
        "複勝": 2,
        "枠連": 3,
        "馬連": 4,
        "ワイド": 5,
        "馬単": 6,
        "三連複": 7,
        "三連単": 8,
    }

    def normalize_kind(kind_text: str) -> Optional[str]:
        # 半角カナ/全角英数/表記ゆれを統一する（例: "ﾜｲﾄﾞ" -> "ワイド", "3連単" -> "3連単"）
        k = unicodedata.normalize("NFKC", kind_text.strip())
        k = re.sub(r"\s+", "", k)
        aliases = {
            "単勝": "単勝",
            "複勝": "複勝",
            "枠連": "枠連",
            "馬連": "馬連",
            "ワイド": "ワイド",
            "馬単": "馬単",
            "三連複": "三連複",
            "三連単": "三連単",
            "3連複": "三連複",
            "3連単": "三連単",
            "ﾜｲﾄﾞ": "ワイド",
        }
        return aliases.get(k)

    def record(kind: str, bet_text: str, pay_text: str, pop: str = "") -> bool:
        bet_nfkc = unicodedata.normalize("NFKC", bet_text)
        nums = [int(x) for x in re.findall(r"\d+", bet_nfkc)]
        key = _normalize_bet_key(kind, nums)
        yen = _parse_yen(pay_text)
        if not key or yen is None:
            return False

        payouts[kind][key] = yen

        bet_display = key.replace("→", " ").replace("-", " ")
        rows_out.append(
            {
                "券種": kind,
                "買い目": bet_display,
                "払戻金": yen,
                "人気": pop.strip(),
                "_order": kind_order.get(kind, 999),
            }
        )
        return True

    def group_size(kind: str) -> Optional[int]:
        if kind in ("単勝", "複勝"):
            return 1
        if kind in ("枠連", "馬連", "ワイド", "馬単"):
            return 2
        if kind in ("三連複", "三連単"):
            return 3
        return None

    current_kind: Optional[str] = None
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if len(cells) < 2:
            continue

        first_text = cells[0].get_text(strip=True)
        detected_kind = normalize_kind(first_text)

        # パターン1: 先頭セルに券種がある（通常）
        if detected_kind is not None:
            current_kind = detected_kind
            kind = detected_kind
            if len(cells) < 3:
                continue
            first_group_start = 1
        # パターン2: rowspan等で券種セルが省略される（続き行）
        elif current_kind is not None:
            kind = current_kind
            first_group_start = 0
        else:
            continue

        # 先頭の (買い目, 払戻, 人気) は「同じセル内に複数行」(複勝/ワイドなど) の場合があるので
        # まずは改行で割って行ベースで処理する。
        bet_cell = cells[first_group_start]
        pay_cell = cells[first_group_start + 1] if len(cells) >= first_group_start + 2 else None
        pop_cell = cells[first_group_start + 2] if len(cells) >= first_group_start + 3 else None
        if pay_cell is None:
            continue

        bet_lines = bet_cell.get_text("\n", strip=True).splitlines()
        pay_lines = pay_cell.get_text("\n", strip=True).splitlines()
        pop_lines: List[str] = pop_cell.get_text("\n", strip=True).splitlines() if pop_cell else []

        succeeded = 0
        if len(bet_lines) == len(pay_lines) and len(pay_lines) > 0:
            for idx, (bet_text, pay_text) in enumerate(zip(bet_lines, pay_lines)):
                pop = pop_lines[idx] if idx < len(pop_lines) else ""
                if record(kind, bet_text, pay_text, pop):
                    succeeded += 1

        # ワイド/複勝などで「買い目セルの改行単位が数字1つずつ」になっていると、
        # bet_lines の1行あたりの数字が足りず record() が失敗します。
        # その場合は、買い目セル全体から数字列を取り出して、券種ごとの桁数で再分割して補完します。
        expected = len(pay_lines)
        size = group_size(kind)
        if size is not None and expected > 0 and succeeded < expected:
            all_nums = [int(x) for x in re.findall(r"\d+", unicodedata.normalize("NFKC", bet_cell.get_text(" ", strip=True)))]
            if len(all_nums) >= size * expected:
                for idx in range(expected):
                    chunk = all_nums[idx * size : (idx + 1) * size]
                    if len(chunk) != size:
                        break
                    bet_text = " ".join(str(n) for n in chunk)
                    pop = pop_lines[idx] if idx < len(pop_lines) else ""
                    record(kind, bet_text, pay_lines[idx], pop)

        # 追加の列がある場合（例: ワイドが1行に3つ並ぶ等）を拾う
        extra_cells = cells[first_group_start + 3 :]
        if extra_cells:
            extra_texts = [c.get_text(" ", strip=True) for c in extra_cells]
            # (買い目, 払戻, 人気) の3列セットとして走査する
            for i in range(0, len(extra_texts), 3):
                bet_text = extra_texts[i] if i < len(extra_texts) else ""
                pay_text = extra_texts[i + 1] if i + 1 < len(extra_texts) else ""
                pop = extra_texts[i + 2] if i + 2 < len(extra_texts) else ""
                if not bet_text or not pay_text:
                    continue
                record(kind, bet_text, pay_text, pop)

    rows_out.sort(key=lambda r: (int(r["_order"]), str(r["券種"]), str(r["買い目"])))
    for r in rows_out:
        r.pop("_order", None)

    return {"payouts": payouts, "rows": rows_out}


def save_json_file(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def try_load_json_file(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None


def normalize_bet_for_lookup(kind_name: str, bet_text: str) -> str:
    """
    購入リスト側の買い目文字列を、払い戻し側のキー形式に合わせて正規化します。
    - 単勝/複勝: "8"
    - 枠連/馬連/ワイド: "8-12"（昇順）
    - 三連複: "2-8-12"（昇順）
    - 馬単/三連単: "8→12" / "8→12→2"（順序ありなのでそのまま）
    """
    bet_nfkc = unicodedata.normalize("NFKC", str(bet_text))
    nums = [int(x) for x in re.findall(r"\d+", bet_nfkc)]
    if kind_name in ("単勝", "複勝"):
        return str(nums[0]) if nums else bet_text
    if kind_name in ("枠連", "馬連", "ワイド"):
        if len(nums) >= 2:
            a, b = sorted(nums[:2])
            return f"{a}-{b}"
        return bet_text
    if kind_name == "三連複":
        if len(nums) >= 3:
            a, b, c = sorted(nums[:3])
            return f"{a}-{b}-{c}"
        return bet_text
    return str(bet_text)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ----------------------------
# Compute tables (same results as main.py)
# ----------------------------
def compute_main_tables(
    odds_json: Dict[str, Any],
) -> Dict[str, List[Dict[str, Any]]]:
    odds = io_json.parse_odds_data(odds_json)
    entries = io_json.parse_tansho_entries(odds)
    market_rows = market_prob.compute_market_win_probabilities(entries)
    prob_by_number = dict(market_prob.build_prob_by_number(market_rows))
    name_by_number = dict(market_prob.build_name_by_number(market_rows))
    all_numbers = [int(e.number) for e in entries]

    # 人気順位（単勝オッズが小さい順 / 同値は馬番昇順）
    sorted_entries = sorted(entries, key=lambda e: (float(e.win_odds), int(e.number)))
    rank_by_number: Dict[int, int] = {int(e.number): i + 1 for i, e in enumerate(sorted_entries)}
    market_table: List[Dict[str, Any]] = []
    for row in market_rows:
        market_table.append(
            {
                "馬番": int(row.number),
                "馬名": row.name,
                "単勝オッズ": float(row.win_odds),
                "人気": int(rank_by_number.get(int(row.number), 0)),
                "市場確率": round(float(row.market_win_prob), 6),
                "市場確率(%)": round(float(row.market_win_prob) * 100, 2),
            }
        )

    # 単勝EV（当選確率=market_win_prob）
    tansho_ev_rows: List[Dict[str, Any]] = []
    for item in odds.tansho_fukusho:
        p = float(prob_by_number[item.number])
        ev = compute_ev(p, float(item.win_odds))
        tansho_ev_rows.append(
            {
                "馬番": int(item.number),
                "馬名": name_by_number.get(item.number, item.name),
                "当選確率(%)": round(p * 100, 2),
                "単勝オッズ": float(item.win_odds),
                "EV": round(ev, 4),
                "EV'": round(compute_ev_prime(p, float(item.win_odds)), 6),
                "p×log(o)": round(compute_p_log_o(p, float(item.win_odds)), 6),
                "Kelly(%)": round(compute_kelly_fraction_clipped(p, float(item.win_odds)) * 100, 2),
            }
        )
    tansho_ev_rows.sort(key=lambda r: r["EV"], reverse=True)

    # 複勝EV（3着以内確率 × 複勝オッズレンジ）
    fukusho_ev_rows: List[Dict[str, Any]] = []
    for item in odds.tansho_fukusho:
        p = ticket_probs.fukusho_probability(prob_by_number, all_numbers, item.number)
        evs = compute_ev_range(p, float(item.place_odds_min), float(item.place_odds_max))
        evs_prime = compute_ev_prime_range(p, float(item.place_odds_min), float(item.place_odds_max))
        kellys = compute_kelly_range(p, float(item.place_odds_min), float(item.place_odds_max))
        p_logs = compute_p_log_o_range(p, float(item.place_odds_min), float(item.place_odds_max))
        fukusho_ev_rows.append(
            {
                "馬番": int(item.number),
                "馬名": name_by_number.get(item.number, item.name),
                "当選確率(%)": round(p * 100, 2),
                "複勝下限": float(item.place_odds_min),
                "複勝上限": float(item.place_odds_max),
                "EV下限": round(float(evs["ev_min"]), 4),
                "EV上限": round(float(evs["ev_max"]), 4),
                "EV'下限": round(float(evs_prime["ev_prime_min"]), 6),
                "EV'上限": round(float(evs_prime["ev_prime_max"]), 6),
                "p×log(o)下限": round(float(p_logs["p_log_o_min"]), 6),
                "p×log(o)上限": round(float(p_logs["p_log_o_max"]), 6),
                "Kelly下限(%)": round(float(kellys["kelly_min"]) * 100, 2),
                "Kelly上限(%)": round(float(kellys["kelly_max"]) * 100, 2),
            }
        )
    fukusho_ev_rows.sort(key=lambda r: r["EV上限"], reverse=True)

    # 馬単EV
    umatan_ev_rows: List[Dict[str, Any]] = []
    for item in odds.umatan:
        p = ticket_probs.umatan_probability(prob_by_number, item.first, item.second)
        ev = compute_ev(p, float(item.odds))
        umatan_ev_rows.append(
            {
                "買い目": f"{item.first}→{item.second}",
                "当選確率(%)": round(p * 100, 3),
                "オッズ": float(item.odds),
                "EV": round(ev, 4),
                "EV'": round(compute_ev_prime(p, float(item.odds)), 6),
                "p×log(o)": round(compute_p_log_o(p, float(item.odds)), 6),
                "Kelly(%)": round(compute_kelly_fraction_clipped(p, float(item.odds)) * 100, 2),
            }
        )
    umatan_ev_rows.sort(key=lambda r: r["EV"], reverse=True)

    # 三連単EV
    sanrentan_ev_rows: List[Dict[str, Any]] = []
    for item in odds.sanrentan:
        p = ticket_probs.sanrentan_probability(prob_by_number, item.first, item.second, item.third)
        ev = compute_ev(p, float(item.odds))
        sanrentan_ev_rows.append(
            {
                "買い目": f"{item.first}→{item.second}→{item.third}",
                "当選確率(%)": round(p * 100, 4),
                "オッズ": float(item.odds),
                "EV": round(ev, 4),
                "EV'": round(compute_ev_prime(p, float(item.odds)), 6),
                "p×log(o)": round(compute_p_log_o(p, float(item.odds)), 6),
                "Kelly(%)": round(compute_kelly_fraction_clipped(p, float(item.odds)) * 100, 2),
            }
        )
    sanrentan_ev_rows.sort(key=lambda r: r["EV"], reverse=True)

    # 馬連EV
    umaren_ev_rows: List[Dict[str, Any]] = []
    for item in odds.umaren:
        p = ticket_probs.umaren_probability(prob_by_number, item.i, item.j)
        ev = compute_ev(p, float(item.odds))
        umaren_ev_rows.append(
            {
                "買い目": f"{item.i}-{item.j}",
                "当選確率(%)": round(p * 100, 3),
                "オッズ": float(item.odds),
                "EV": round(ev, 4),
                "EV'": round(compute_ev_prime(p, float(item.odds)), 6),
                "p×log(o)": round(compute_p_log_o(p, float(item.odds)), 6),
                "Kelly(%)": round(compute_kelly_fraction_clipped(p, float(item.odds)) * 100, 2),
            }
        )
    umaren_ev_rows.sort(key=lambda r: r["EV"], reverse=True)

    # 三連複EV
    sanrenpuku_ev_rows: List[Dict[str, Any]] = []
    for item in odds.sanrenpuku:
        p = ticket_probs.sanrenpuku_probability(prob_by_number, item.a, item.b, item.c)
        ev = compute_ev(p, float(item.odds))
        sanrenpuku_ev_rows.append(
            {
                "買い目": f"{item.a}-{item.b}-{item.c}",
                "当選確率(%)": round(p * 100, 4),
                "オッズ": float(item.odds),
                "EV": round(ev, 4),
                "EV'": round(compute_ev_prime(p, float(item.odds)), 6),
                "p×log(o)": round(compute_p_log_o(p, float(item.odds)), 6),
                "Kelly(%)": round(compute_kelly_fraction_clipped(p, float(item.odds)) * 100, 2),
            }
        )
    sanrenpuku_ev_rows.sort(key=lambda r: r["EV"], reverse=True)

    # ワイドEV（レンジ）
    wide_ev_rows: List[Dict[str, Any]] = []
    for item in odds.wide:
        p = ticket_probs.wide_probability(prob_by_number, all_numbers, item.i, item.j)
        evs = compute_ev_range(p, float(item.odds_min), float(item.odds_max))
        evs_prime = compute_ev_prime_range(p, float(item.odds_min), float(item.odds_max))
        kellys = compute_kelly_range(p, float(item.odds_min), float(item.odds_max))
        p_logs = compute_p_log_o_range(p, float(item.odds_min), float(item.odds_max))
        wide_ev_rows.append(
            {
                "買い目": f"{item.i}-{item.j}",
                "当選確率(%)": round(p * 100, 3),
                "下限": float(item.odds_min),
                "上限": float(item.odds_max),
                "EV下限": round(float(evs["ev_min"]), 4),
                "EV上限": round(float(evs["ev_max"]), 4),
                "EV'下限": round(float(evs_prime["ev_prime_min"]), 6),
                "EV'上限": round(float(evs_prime["ev_prime_max"]), 6),
                "p×log(o)下限": round(float(p_logs["p_log_o_min"]), 6),
                "p×log(o)上限": round(float(p_logs["p_log_o_max"]), 6),
                "Kelly下限(%)": round(float(kellys["kelly_min"]) * 100, 2),
                "Kelly上限(%)": round(float(kellys["kelly_max"]) * 100, 2),
            }
        )
    wide_ev_rows.sort(key=lambda r: r["EV上限"], reverse=True)

    return {
        "market": market_table,
        "tansho_ev": tansho_ev_rows,
        "fukusho_ev": fukusho_ev_rows,
        "umatan_ev": umatan_ev_rows,
        "sanrentan_ev": sanrentan_ev_rows,
        "umaren_ev": umaren_ev_rows,
        "sanrenpuku_ev": sanrenpuku_ev_rows,
        "wide_ev": wide_ev_rows,
    }


def _percentile(values: List[float], p: float) -> float:
    # p in [0,1]
    xs = sorted(v for v in values if v == v)  # drop NaN
    if not xs:
        return float("nan")
    if p <= 0:
        return xs[0]
    if p >= 1:
        return xs[-1]
    idx = int(round(p * (len(xs) - 1)))
    return xs[idx]


def stake_from_budget(budget_yen: int, kelly_fraction: float) -> int:
    """
    賭け金ルール（ユーザー指定）:
      - 条件で抽出した KS を使い、(KS*100)/4 % を賭ける
        -> 予算比で言うと (KS/4) を掛ける
      - (KS/4) が 5% 以上なら、その賭け率を 1/2 倍（張り過ぎ防止）
      - 馬券は100円単位
      - 算出額が min_round_up_yen 未満なら 0円
      - min_round_up_yen〜99円なら 100円
      - 100円以上は 100円単位で切り捨て
    """
    if budget_yen <= 0:
        return 0
    if kelly_fraction <= 0:
        return 0

    bet_fraction = compute_bet_fraction(float(kelly_fraction))
    return compute_stake_from_budget_fraction(budget_yen, bet_fraction)


def compute_bet_fraction(ks: float) -> float:
    """
    KS から「予算に対する賭け割合（0〜1）」を作ります。

    仕様:
      - ベース: bet_fraction = KS/4
      - bet_fraction が 5% 以上なら 1/2 倍
    """
    bet_fraction = float(ks) / 4.0
    if bet_fraction >= 0.05:
        bet_fraction *= 0.5
    return bet_fraction


def compute_stake_from_budget_fraction(
    budget_yen: int,
    bet_fraction: float,
    min_round_up_yen: int = 80,
    max_stake_per_bet_yen: int = 0,
) -> int:
    """
    予算と賭け割合から、馬券の購入額（円）を決めます（100円単位）。
    """
    if budget_yen <= 0:
        return 0
    if bet_fraction <= 0:
        return 0

    stake_raw = float(budget_yen) * float(bet_fraction)
    if stake_raw < float(min_round_up_yen):
        return 0
    if stake_raw < 100.0:
        return 100
    stake = int(stake_raw // 100) * 100
    if max_stake_per_bet_yen > 0:
        stake = min(stake, int(max_stake_per_bet_yen))
        stake = int(stake // 100) * 100
    return stake


def compute_bet_fraction_for_kind(kind: str, ks: float) -> float:
    """
    券種ごとの最終的な賭け割合（予算比）を返します。

    追加仕様:
      - 複勝は、計算結果の掛け金をさらに半分にする（= 賭け割合も 1/2）
      - ワイドも、計算結果の掛け金をさらに半分にする（= 賭け割合も 1/2）
    """
    base = compute_bet_fraction(ks)
    if kind in ("複勝", "ワイド"):
        return base * 0.5
    return base


def _desired_units_from_budget_fraction(
    budget_yen: int,
    bet_fraction: float,
    min_round_up_yen: int = 80,
) -> int:
    """
    予算と賭け割合から、希望する購入単位（100円=1）を作る。
    min_round_up_yen〜99円は100円に切り上げ（=1 unit）。
    """
    if budget_yen <= 0:
        return 0
    if bet_fraction <= 0:
        return 0
    stake_raw = float(budget_yen) * float(bet_fraction)
    if stake_raw < float(min_round_up_yen):
        return 0
    if stake_raw < 100.0:
        return 1
    return int(stake_raw // 100)


def allocate_stakes_with_budget_cap(
    budget_yen: int,
    bet_fractions: List[float],
    min_round_up_yen: int = 80,
    max_stake_per_bet_yen: int = 0,
) -> List[int]:
    """
    bet_fraction を各買い目にそのまま適用すると、合計が予算を超えることがあるため、
    合計が予算（100円単位）を超えないように下位（リスト後方）から削る。
    """
    budget_units = int(budget_yen) // 100
    max_units_per_bet = 0
    if int(max_stake_per_bet_yen) > 0:
        max_units_per_bet = int(max_stake_per_bet_yen) // 100

    desired_units: List[int] = []
    for f in bet_fractions:
        u = _desired_units_from_budget_fraction(budget_yen, f, min_round_up_yen=min_round_up_yen)
        if max_units_per_bet > 0:
            u = min(u, max_units_per_bet)
        desired_units.append(u)
    total_units = sum(desired_units)
    if total_units <= budget_units:
        return [u * 100 for u in desired_units]

    extra = total_units - budget_units
    units = desired_units[:]
    for idx in range(len(units) - 1, -1, -1):
        while extra > 0 and units[idx] > 0:
            units[idx] -= 1
            extra -= 1
        if extra <= 0:
            break

    return [u * 100 for u in units]


def rows_to_tsv(rows: List[Dict[str, Any]]) -> str:
    # 表（list[dict]）をコピーしやすいTSVに変換
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = ["\t".join(headers)]
    for row in rows:
        lines.append("\t".join(str(row.get(h, "")) for h in headers))
    return "\n".join(lines)


def render_copy_button(label: str, text: str, key: str) -> None:
    # クリップボードへコピーするボタン（Streamlit標準には無いのでHTML/JSで実装）
    payload = json.dumps(text, ensure_ascii=False)
    html = f"""
    <div style="display:flex; gap:8px; align-items:center; margin:4px 0 8px 0;">
      <button id="copy-btn-{key}" style="padding:6px 10px; border-radius:6px; border:1px solid #ccc; background:#fff; cursor:pointer;">
        {label}
      </button>
      <span id="copy-msg-{key}" style="color:#666; font-size:12px;"></span>
    </div>
    <script>
      const btn = document.getElementById("copy-btn-{key}");
      const msg = document.getElementById("copy-msg-{key}");
      const text = {payload};
      btn.addEventListener("click", async () => {{
        try {{
          await navigator.clipboard.writeText(text);
          msg.textContent = "copied";
          setTimeout(() => msg.textContent = "", 1200);
        }} catch (e) {{
          msg.textContent = "copy failed";
          setTimeout(() => msg.textContent = "", 2000);
        }}
      }});
    </script>
    """
    components.html(html, height=48)


def build_recommendations(
    tables: Dict[str, List[Dict[str, Any]]],
    ev_and_min: float = 0.0,
    prob_and_min: float = 0.0,
    prob_or_min: float = 0.0,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    抽出に使う指標（safe版）:
      - KS（Kelly_safe） = 0.60*Kelly_low + 0.40*Kelly_up
      - EVS（EV_safe）   = 0.60*EV_low    + 0.40*EV_up

    参考（レンジが無い券種は low=up とみなす）
      - EV_low/EV_up: EV下限/EV上限
      - Kelly_low/Kelly_up: Kelly下限/Kelly上限

    LS は (p×log(o)) のスコア（レンジがある場合は合成）:
      - LL = 下限, LU = 上限
      - LS = 0.30*LL + 0.70*LU
      - レンジが無い券種は LL=LU とみなす

    条件（新）:
      - (EVU >= ev_and_min かつ P >= prob_and_min) または
      - (P >= prob_or_min)
    """

    candidates: List[Dict[str, Any]] = []

    def safe_value(low: float, up: float) -> float:
        return 0.60 * float(low) + 0.40 * float(up)

    def calc_ls(ll: float, lu: float) -> float:
        return 0.30 * float(ll) + 0.70 * float(lu)

    def add_single(kind: str, bet: str, row: Dict[str, Any]) -> None:
        kelly_fraction = float(row.get("Kelly(%)", 0.0)) / 100.0
        ev_value = float(row["EV"])
        ev_u = ev_value
        prob = float(row.get("当選確率(%)", 0.0)) / 100.0
        ll_value = float(row.get("p×log(o)", 0.0))
        lu_value = ll_value
        ls_score = calc_ls(ll_value, lu_value)
        candidates.append(
            {
                "種別": kind,
                "買い目": bet,
                "EVS": ev_value,
                "EVU": ev_u,
                "P": prob,
                "KS": kelly_fraction,
                "LL": ll_value,
                "LU": lu_value,
                "LS": ls_score,
            }
        )

    def add_range(kind: str, bet: str, row: Dict[str, Any]) -> None:
        ev_low = float(row["EV下限"])
        ev_up = float(row["EV上限"])
        ev_u = ev_up
        prob = float(row.get("当選確率(%)", 0.0)) / 100.0
        kelly_low = float(row.get("Kelly下限(%)", 0.0)) / 100.0
        kelly_up = float(row.get("Kelly上限(%)", 0.0)) / 100.0
        ll_value = float(row.get("p×log(o)下限", 0.0))
        lu_value = float(row.get("p×log(o)上限", ll_value))
        ls_score = calc_ls(ll_value, lu_value)
        candidates.append(
            {
                "種別": kind,
                "買い目": bet,
                "EVS": safe_value(ev_low, ev_up),
                "EVU": ev_u,
                "P": prob,
                "KS": safe_value(kelly_low, kelly_up),
                "LL": ll_value,
                "LU": lu_value,
                "LS": ls_score,
            }
        )

    for row in tables.get("tansho_ev", []):
        add_single("単勝", str(row["馬番"]), row)
    for row in tables.get("fukusho_ev", []):
        add_range("複勝", str(row["馬番"]), row)
    for row in tables.get("umatan_ev", []):
        add_single("馬単", str(row["買い目"]), row)
    for row in tables.get("sanrentan_ev", []):
        add_single("三連単", str(row["買い目"]), row)
    for row in tables.get("umaren_ev", []):
        add_single("馬連", str(row["買い目"]), row)
    for row in tables.get("sanrenpuku_ev", []):
        add_single("三連複", str(row["買い目"]), row)
    for row in tables.get("wide_ev", []):
        add_range("ワイド", str(row["買い目"]), row)

    filtered: List[Dict[str, Any]] = []
    for c in candidates:
        evu = float(c.get("EVU", 0.0))
        p = float(c.get("P", 0.0))
        pass_and = (evu >= float(ev_and_min)) and (p >= float(prob_and_min))
        pass_p_only = p >= float(prob_or_min)
        if pass_and or pass_p_only:
            c2 = dict(c)
            c2["_pass_and"] = pass_and
            c2["_pass_p_only"] = pass_p_only
            filtered.append(c2)
    filtered.sort(
        key=lambda c: (float(c.get("EVU", 0.0)), float(c.get("LS", 0.0))),
        reverse=True,
    )
    return filtered, float("nan")


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="競馬オッズ → JSON/EV", layout="wide")
st.title("競馬計算アプリ")

_DEFAULT_ODDS_URL = "https://nar.netkeiba.com/odds/?race_id=202644011601&type=b0"
_DEFAULT_PAYOUT_URL = "https://nar.netkeiba.com/race/result.html?race_id=202655012404"

if "odds_url" not in st.session_state:
    st.session_state["odds_url"] = _DEFAULT_ODDS_URL
if "payout_url" not in st.session_state:
    st.session_state["payout_url"] = _DEFAULT_PAYOUT_URL

if "user_bets" not in st.session_state:
    # 各券種ごとに「馬券そのものの人気順位（整数）」を保存する
    st.session_state["user_bets"] = {"単勝": [], "枠連": [], "馬単": [], "馬連": [], "三連複": []}


def _on_bet_change(kind: str, raw_key: str) -> None:
    # 「馬券そのものの人気順位」を保持（例: 馬単 1人気,2人気 → [1,2]）
    bets = parse_user_ticket_popularity_ranks(str(st.session_state.get(raw_key, "") or ""))
    user_bets = dict(st.session_state.get("user_bets", {}) or {})
    user_bets[kind] = bets
    st.session_state["user_bets"] = user_bets


def _clear_text_input(key: str) -> None:
    # Streamlit の widget(key=...) は「生成後に同じrun内で session_state[key] を直接代入」すると例外になる。
    # on_click コールバックで更新するのが安全。
    st.session_state[key] = ""


def _on_output_click() -> None:
    """
    「出力」ボタンの on_click。
    - 入力URLを一旦退避
    - 入力欄をクリア
    - 実処理フラグを立てる（次のrunで実行）
    """
    st.session_state["_run_output"] = True
    st.session_state["_submitted_mode"] = str(st.session_state.get("ui_mode", "") or "")
    st.session_state["_submitted_odds_url"] = str(st.session_state.get("odds_url", "") or "")
    st.session_state["_submitted_payout_url"] = str(st.session_state.get("payout_url", "") or "")
    st.session_state["_submitted_payout_multi"] = bool(st.session_state.get("payout_multi", False))
    st.session_state["_submitted_payout_max_round"] = int(st.session_state.get("payout_max_round", 12))
    st.session_state["_submitted_scan_year"] = int(st.session_state.get("scan_year", 0))
    st.session_state["_submitted_scan_pp"] = str(st.session_state.get("scan_pp", "") or "")
    st.session_state["_submitted_scan_cc_start"] = int(st.session_state.get("scan_cc_start", 1))
    st.session_state["_submitted_scan_cc_max"] = int(st.session_state.get("scan_cc_max", 12))
    st.session_state["_submitted_scan_dd_start"] = int(st.session_state.get("scan_dd_start", 1))
    st.session_state["_submitted_scan_dd_max"] = int(st.session_state.get("scan_dd_max", 12))
    st.session_state["_submitted_scan_rr_max"] = int(st.session_state.get("scan_rr_max", 12))
    st.session_state["_submitted_scan_total_limit"] = int(st.session_state.get("scan_total_limit", 200))
    st.session_state["_submitted_nar_scan_year"] = int(st.session_state.get("nar_scan_year", 0))
    st.session_state["_submitted_nar_scan_pp"] = str(st.session_state.get("nar_scan_pp", "") or "")
    st.session_state["_submitted_nar_scan_mm_min"] = int(st.session_state.get("nar_scan_mm_min", 1))
    st.session_state["_submitted_nar_scan_mm_max"] = int(st.session_state.get("nar_scan_mm_max", 12))
    st.session_state["_submitted_nar_scan_dd_min"] = int(st.session_state.get("nar_scan_dd_min", 1))
    st.session_state["_submitted_nar_scan_dd_max"] = int(st.session_state.get("nar_scan_dd_max", 31))
    st.session_state["_submitted_nar_scan_rr_max"] = int(st.session_state.get("nar_scan_rr_max", 12))
    st.session_state["_submitted_nar_scan_total_limit"] = int(st.session_state.get("nar_scan_total_limit", 200))
    st.session_state["odds_url"] = ""
    st.session_state["payout_url"] = ""
    # 前回結果をクリア（次のrunで新しい結果だけを表示する）
    for k in ("_last_error",):
        if k in st.session_state:
            del st.session_state[k]

    # 直前にエンターを押していないケースでも、クリック時点の値を取り込む
    for kind, raw_key in (
        ("単勝", "bet_raw_tansho"),
        ("枠連", "bet_raw_wakuren"),
        ("馬単", "bet_raw_umatan"),
        ("馬連", "bet_raw_umaren"),
        ("三連複", "bet_raw_sanrenpuku"),
    ):
        _on_bet_change(kind, raw_key)


def _on_reset_results_click() -> None:
    # 前回の出力状態をクリアして、入力し直して再実行できるようにする
    for k in (
        "_run_output",
        "_submitted_mode",
        "_submitted_odds_url",
        "_submitted_payout_url",
        "_submitted_payout_multi",
        "_submitted_payout_max_round",
        "_submitted_scan_year",
        "_submitted_scan_pp",
        "_submitted_scan_cc_start",
        "_submitted_scan_cc_max",
        "_submitted_scan_dd_start",
        "_submitted_scan_dd_max",
        "_submitted_scan_rr_max",
        "_submitted_scan_total_limit",
        "_submitted_nar_scan_year",
        "_submitted_nar_scan_pp",
        "_submitted_nar_scan_mm_min",
        "_submitted_nar_scan_mm_max",
        "_submitted_nar_scan_dd_min",
        "_submitted_nar_scan_dd_max",
        "_submitted_nar_scan_rr_max",
        "_submitted_nar_scan_total_limit",
        "_last_error",
    ):
        if k in st.session_state:
            del st.session_state[k]

    # 買い目入力は残す（必要ならここで消す）


ui_mode = st.radio(
    "モード",
    [
        "払戻→Sheets（IDスキャン）",
        "払戻→Sheets（IDスキャン）（地方競馬版）",
        "ジニ係数算出（オッズURL）",
    ],
    horizontal=True,
    key="ui_mode",
)

# 互換のための既定値（現UIでは非表示）
payout_url = str(st.session_state.get("payout_url", "") or "")
budget_yen = 10000
save_odds_local = True
save_payouts_local = True
hide_zero_stake = True
pick_mode = "通常（条件式で抽出）"
ev_and_threshold = 0.0
prob_threshold_percent = 0.0
prob_or_threshold_percent = 0.0
min_round_up_yen = 80

if ui_mode == "払戻→Sheets（IDスキャン）":
    st.caption("race_id の構成: 年(YYYY) / 馬場ID(PP) / 開催回(CC) / 開催日目(DD) / レース番号(RR)")
    st.caption("例: 202606010301 = 2026年 / 馬場ID06(中山) / 1回 / 3日目 / 1R")
    st.caption(
        "中央競馬の馬場ID: "
        + " / ".join([f"{k:02d}:{v}" for k, v in sorted(JRA_PLACE_BY_CODE.items())])
    )
    col_s1, col_s2, col_s3, col_s4, col_s5 = st.columns([1, 1, 1, 1, 1])
    with col_s1:
        st.number_input("年(YYYY)", min_value=1990, max_value=2100, value=2026, step=1, key="scan_year")
    with col_s2:
        st.text_input("馬場ID(PP 2桁)", value="06", key="scan_pp")
    with col_s3:
        st.number_input("開催回 開始(CC)", min_value=1, max_value=99, value=1, step=1, key="scan_cc_start")
    with col_s4:
        st.number_input(
            "開催日目 開始(DD)", min_value=1, max_value=99, value=1, step=1, key="scan_dd_start"
        )
    with col_s5:
        st.number_input("レース番号 最大(RR)", min_value=1, max_value=24, value=12, step=1, key="scan_rr_max")

    col_s6, col_s7, col_s8 = st.columns([1, 1, 1])
    with col_s6:
        st.number_input("開催回 最大(CC)", min_value=1, max_value=99, value=12, step=1, key="scan_cc_max")
    with col_s7:
        st.number_input("開催日目 最大(DD)", min_value=1, max_value=99, value=12, step=1, key="scan_dd_max")
    with col_s8:
        st.number_input(
            "最大取得レース数",
            min_value=1,
            max_value=1000,
            value=200,
            step=10,
            key="scan_total_limit",
            help="安全のための上限です。",
        )

if ui_mode == "払戻→Sheets（IDスキャン）（地方競馬版）":
    st.caption("race_id の構成: 年(YYYY) / 馬場ID(PP) / 月(MM) / 日(DD) / レース番号(RR)")
    st.caption("例: 202665013101 = 2026年 / 馬場ID65 / 1月 / 31日 / 1R")
    st.caption(
        "地方競馬(主要)の馬場ID: "
        + " / ".join([f"{k:02d}:{v}" for k, v in sorted(NAR_PLACE_BY_CODE.items())])
    )
    nar_pp_now = re.sub(r"[^0-9]", "", str(st.session_state.get("nar_scan_pp", "") or "")).zfill(2)[:2]
    nar_pp_name = place_name_from_code_any(parse_int_maybe(nar_pp_now))
    st.caption(
        f"入力中の馬場ID {nar_pp_now}: "
        + (nar_pp_name if nar_pp_name else "（名称は取得時に自動判定）")
    )
    col_n1, col_n2, col_n3, col_n4, col_n5 = st.columns([1, 1, 1, 1, 1])
    with col_n1:
        st.number_input("年(YYYY)", min_value=1990, max_value=2100, value=2026, step=1, key="nar_scan_year")
    with col_n2:
        st.text_input("馬場ID(PP 2桁)", value="65", key="nar_scan_pp")
    with col_n3:
        st.number_input("月 最大(MM)", min_value=1, max_value=12, value=12, step=1, key="nar_scan_mm_max")
    with col_n4:
        st.number_input("日 最大(DD)", min_value=1, max_value=31, value=31, step=1, key="nar_scan_dd_max")
    with col_n5:
        st.number_input("レース番号 最大(RR)", min_value=1, max_value=24, value=12, step=1, key="nar_scan_rr_max")

    col_n6, col_n7, col_n8 = st.columns([1, 1, 1])
    with col_n6:
        st.number_input("月 最小(MM)", min_value=1, max_value=12, value=1, step=1, key="nar_scan_mm_min")
    with col_n7:
        st.number_input("日 最小(DD)", min_value=1, max_value=31, value=1, step=1, key="nar_scan_dd_min")
    with col_n8:
        st.number_input(
            "最大取得レース数",
            min_value=1,
            max_value=1000,
            value=200,
            step=10,
            key="nar_scan_total_limit",
            help="安全のための上限です。",
        )

if ui_mode == "ジニ係数算出（オッズURL）":
    st.caption("単勝オッズURLを入力し、単勝オッズ上位半分でジニ係数を算出します。")
    st.text_input(
        "オッズURL",
        key="odds_url",
        help="race_id を含むURLを推奨（例: https://nar.netkeiba.com/odds/?race_id=...&type=b0）",
    )

st.caption("レポート設定（Google Sheets）")
sheets_enabled = st.checkbox("結果をGoogle Sheetsに記録", value=True)
sheets_spreadsheet_id = st.text_input(
    "Spreadsheet ID",
    value="1AdgLnjPhweDaVKB-QO1Sd4iXGozs2T_Nw2OBSNYmLEY",
    disabled=(not sheets_enabled),
)
sheets_sheet_name = st.text_input("シート名（予想モード互換）", value="result", disabled=(not sheets_enabled))
st.caption(
    "払戻→Sheetsの追記列: A=人気 / B=金額 / C=場所コード / D=ジニ係数 / E=出走頭数"
    "（加えてレース単位のジニ係数はシート『ジニ係数』にも追記）"
)

# このフォルダ内にサービスアカウントJSONを置いて使えるようにする（アップロード不要）
default_sa_path = ""
try:
    candidates = [p.name for p in Path(".").glob("*.json") if p.name not in ("odds.json", "payouts.json")]
    default_sa_path = "service_account.json" if Path("service_account.json").exists() else ""
    if not default_sa_path and candidates:
        default_sa_path = candidates[0]
except Exception:
    default_sa_path = ""
sheets_sa_path = st.text_input(
    "サービスアカウントJSON（ファイル名 / 同フォルダ）",
    value=default_sa_path,
    help="例: service_account.json / black-practice-xxxx.json",
    disabled=(not sheets_enabled),
)
sheets_sa_file = st.file_uploader(
    "サービスアカウントJSON（credentials）",
    type=["json"],
    disabled=(not sheets_enabled),
)

# UI表示ルールにより odds.json の内容は表示しない（取得・保存は内部で維持）

col_run1, col_run2 = st.columns([1, 1])
with col_run1:
    st.button("出力", on_click=_on_output_click)
with col_run2:
    st.button("結果をリセット", on_click=_on_reset_results_click)

if st.session_state.get("_run_output"):
    try:
        # st.stop() が finally を飛ばすケースに備えて、先に落としておく（実行はこのrun内だけ）
        st.session_state["_run_output"] = False
        mode_run = str(st.session_state.get("_submitted_mode", "") or "")
        odds_url_run = str(st.session_state.get("_submitted_odds_url", "") or "")
        payout_url = str(st.session_state.get("_submitted_payout_url", "") or "")
        payout_multi = bool(st.session_state.get("_submitted_payout_multi", False))
        payout_max_round = int(st.session_state.get("_submitted_payout_max_round", 12))
        scan_year = int(st.session_state.get("_submitted_scan_year", 0))
        scan_pp = str(st.session_state.get("_submitted_scan_pp", "") or "")
        scan_cc_start = int(st.session_state.get("_submitted_scan_cc_start", 1))
        scan_cc_max = int(st.session_state.get("_submitted_scan_cc_max", 12))
        scan_dd_start = int(st.session_state.get("_submitted_scan_dd_start", 1))
        scan_dd_max = int(st.session_state.get("_submitted_scan_dd_max", 12))
        scan_rr_max = int(st.session_state.get("_submitted_scan_rr_max", 12))
        scan_total_limit = int(st.session_state.get("_submitted_scan_total_limit", 200))
        nar_scan_year = int(st.session_state.get("_submitted_nar_scan_year", 0))
        nar_scan_pp = str(st.session_state.get("_submitted_nar_scan_pp", "") or "")
        nar_scan_mm_min = int(st.session_state.get("_submitted_nar_scan_mm_min", 1))
        nar_scan_mm_max = int(st.session_state.get("_submitted_nar_scan_mm_max", 12))
        nar_scan_dd_min = int(st.session_state.get("_submitted_nar_scan_dd_min", 1))
        nar_scan_dd_max = int(st.session_state.get("_submitted_nar_scan_dd_max", 31))
        nar_scan_rr_max = int(st.session_state.get("_submitted_nar_scan_rr_max", 12))
        nar_scan_total_limit = int(st.session_state.get("_submitted_nar_scan_total_limit", 200))

        if mode_run == "ジニ係数算出（オッズURL）":
            odds_url = str(odds_url_run or "").strip()
            if not odds_url:
                raise ValueError("オッズURLを入力してください。")

            status = st.status("処理中…（ジニ係数算出）", expanded=False)
            race_id = ""
            try:
                qs = parse_qs(urlparse(odds_url).query)
                race_id = str(qs.get("race_id", [""])[0] or "")
            except Exception:
                race_id = ""

            race_ctx: Dict[str, Any] = {"race_id": race_id, "date": "", "venue": "", "race_no": None}
            runner_count: Optional[int] = None
            win_rows: List[Dict[str, Any]] = []
            if race_id:
                try:
                    status.update(label=f"処理中…（単勝オッズ） race_id={race_id}", state="running")
                    win_rows = fetch_win_rows_for_race(race_id)
                except Exception:
                    win_rows = []

            html = ""
            if (not win_rows) or (not race_ctx.get("date") and not race_ctx.get("venue")):
                try:
                    status.update(label="処理中…（URL取得）", state="running")
                    html = fetch_html(odds_url)
                    race_ctx = parse_race_context_from_html(odds_url, html)
                    if not race_id:
                        race_id = str(race_ctx.get("race_id", "") or "")
                    runner_count = parse_runner_count_from_result_html(html)
                except Exception:
                    html = ""

            if not win_rows and race_id:
                try:
                    status.update(label=f"処理中…（単勝オッズ） race_id={race_id}", state="running")
                    win_rows = fetch_win_rows_for_race(race_id)
                except Exception:
                    win_rows = []

            if not win_rows:
                if html:
                    win_map = parse_win_odds_from_html_generic(html)
                    win_rows = compute_win_odds_rows_from_map(win_map)

            if not win_rows:
                raise RuntimeError(
                    "単勝オッズを取得できませんでした。"
                    "race_id を含むURLを入力し、時間を置いて再試行してください（403/429時は取得不可）。"
                )

            if race_id and not str(race_ctx.get("venue", "") or ""):
                place_code = parse_place_code_from_race_id(race_id)
                race_ctx["venue"] = place_name_from_code_any(place_code)

            cnt = count_valid_win_rows(win_rows)
            if runner_count is None and cnt is not None:
                runner_count = int(cnt)

            gini_row = compute_race_gini_from_win_rows(
                race_id if race_id else "unknown",
                win_rows,
                runner_count_hint=runner_count,
            )
            if gini_row is None:
                raise RuntimeError("ジニ係数を計算できませんでした（有効な単勝オッズが不足しています）。")

            valid_rows: List[Dict[str, Any]] = []
            for r in win_rows:
                if not isinstance(r, dict):
                    continue
                try:
                    no = int(r.get("馬番", 0))
                    o = float(r.get("単勝オッズ", 0.0))
                except Exception:
                    continue
                if no <= 0 or o <= 0:
                    continue
                valid_rows.append({"馬番": int(no), "単勝オッズ": float(o)})
            valid_rows.sort(key=lambda x: (float(x["単勝オッズ"]), int(x["馬番"])))
            for i, row in enumerate(valid_rows, start=1):
                row["人気"] = int(i)

            top_count = int(gini_row["ジニ対象頭数"])
            top_half_rows = valid_rows[:top_count]

            race_parts: List[str] = []
            if race_ctx.get("date"):
                race_parts.append(str(race_ctx["date"]))
            if race_ctx.get("venue"):
                race_parts.append(str(race_ctx["venue"]))
            if race_ctx.get("race_no") is not None:
                race_parts.append(f'{int(race_ctx["race_no"])}R')
            if race_id:
                race_parts.append(f"(race_id={race_id})")
            if race_parts:
                st.caption("レース情報: " + " ".join(race_parts))

            st.subheader("ジニ係数（単勝オッズ上位半分）")
            col_g1, col_g2, col_g3 = st.columns(3)
            col_g1.metric("ジニ係数", f'{float(gini_row["ジニ係数"]):.6f}')
            col_g2.metric("出走頭数", str(int(gini_row["出走頭数"])))
            col_g3.metric("ジニ対象頭数", str(int(gini_row["ジニ対象頭数"])))

            st.subheader("ジニ対象馬（上位半分）")
            render_copy_button("ジニ対象馬をコピー(TSV)", rows_to_tsv(top_half_rows), key="gini_top_half_rows")
            st.dataframe(top_half_rows, use_container_width=True)

            st.subheader("単勝オッズ一覧（有効値）")
            render_copy_button("単勝オッズ一覧をコピー(TSV)", rows_to_tsv(valid_rows), key="gini_all_win_rows")
            st.dataframe(valid_rows, use_container_width=True)

            status.update(label="完了", state="complete")
            st.session_state["_run_output"] = False
            st.stop()

        if mode_run == "払戻→Sheets（IDスキャン）（地方競馬版）":
            pp = re.sub(r"[^0-9]", "", nar_scan_pp).zfill(2)[:2]
            if not (nar_scan_year and pp and len(pp) == 2):
                raise ValueError("年(YYYY) と 馬場ID(PP 2桁) を指定してください。")

            allowed_kinds = {"単勝", "枠連", "馬単", "馬連", "三連複"}
            sheet_by_kind = {
                "単勝": "単勝大好き",
                "枠連": "枠連大好き",
                "馬単": "馬単大好き",
                "馬連": "馬連大好き",
                "三連複": "三連複大好き",
            }

            def mk_race_id(year: int, pp2: str, mm: int, dd: int, rr: int) -> str:
                return f"{year:04d}{pp2}{mm:02d}{dd:02d}{rr:02d}"

            def result_url_for(race_id: str) -> str:
                return f"https://nar.netkeiba.com/race/result.html?race_id={race_id}"

            payout_rows: List[Dict[str, Any]] = []
            gini_rows_all: List[Dict[str, Any]] = []
            fetched_races = 0
            stopped_reason = ""

            status = st.status("処理中…（地方IDスキャン）", expanded=False)
            progress = st.progress(0.0)

            settlement_kinds = ("単勝", "枠連", "馬単", "馬連", "三連複")
            user_bets = st.session_state.get("user_bets", {}) or {}
            settle_race_count = 0
            settle_stake_total = 0
            settle_return_total = 0
            settle_stake_by_kind: Dict[str, int] = {k: 0 for k in settlement_kinds}
            settle_return_by_kind: Dict[str, int] = {k: 0 for k in settlement_kinds}
            settle_hit_details: Dict[str, List[str]] = {k: [] for k in settlement_kinds}

            # MM/DD は「最大→最小」で走査
            for mm in range(int(nar_scan_mm_max), int(nar_scan_mm_min) - 1, -1):
                for dd in range(int(nar_scan_dd_max), int(nar_scan_dd_min) - 1, -1):
                    rr_valid_any = False
                    for rr in range(1, max(1, int(nar_scan_rr_max)) + 1):
                        race_id = mk_race_id(int(nar_scan_year), pp, mm, dd, rr)
                        try:
                            status.update(label=f"処理中…（取得） race_id={race_id}", state="running")
                            result_url = result_url_for(race_id)
                            html = fetch_html(result_url)
                            if looks_like_invalid_result_page(html):
                                raise RuntimeError(f"Invalid/empty result page: {result_url}")
                            bundle = parse_payouts_bundle_from_result_html(html)
                        except Exception:
                            if rr == 1:
                                # この開催日にはレースがない
                                break
                            # この開催日のRRが尽きた
                            break

                        # 「払戻がある=実施済みレース」だけを対象にする
                        if not result_bundle_has_any_payout(bundle, html):
                            if rr == 1:
                                # この開催日にはレースがない/未確定
                                break
                            # この開催日のRRが尽きた/未確定
                            break

                        race_ctx = parse_race_context_from_html(result_url, html)
                        place_code = parse_place_code_from_race_id(race_id)
                        place_name = str(race_ctx.get("venue", "") or "") or place_name_from_code_any(place_code)
                        runner_count = parse_runner_count_from_result_html(html)
                        gini_value: Optional[float] = None
                        gini_target_count: Optional[int] = None
                        gini_row: Optional[Dict[str, Any]] = None

                        rr_valid_any = True
                        fetched_races += 1
                        if int(nar_scan_total_limit) > 0:
                            progress.progress(min(1.0, float(fetched_races) / float(nar_scan_total_limit)))

                        # 収支（買い目人気）をレース単位で集計（実際に取得できた回数だけ加算）
                        if any(user_bets.get(k) for k in settlement_kinds):
                            per_race = compute_settlement_from_payout_rows(
                                payout_rows=bundle.get("rows", []) if isinstance(bundle, dict) else [],
                                user_bets=user_bets,
                                kinds=settlement_kinds,
                                stake_per_bet_yen=100,
                            )
                            settle_race_count += 1
                            settle_stake_total += int(per_race["stake_total"])
                            settle_return_total += int(per_race["returned_total"])
                            for k in settlement_kinds:
                                settle_stake_by_kind[k] += int(per_race["stake_by_kind"].get(k, 0))
                                settle_return_by_kind[k] += int(per_race["returned_by_kind"].get(k, 0))
                                for d in per_race["hit_details"].get(k, []):
                                    settle_hit_details[k].append(f"{race_id} {d}")

                        # 同時に単勝オッズを取得し、上位半分でジニ係数を計算
                        try:
                            status.update(label=f"処理中…（オッズ） race_id={race_id}", state="running")
                            win_rows = fetch_win_rows_for_race(race_id)
                            if not win_rows:
                                win_rows = parse_win_odds_rows_from_result_html(html)
                            cnt = count_valid_win_rows(win_rows)
                            if runner_count is None and cnt is not None:
                                runner_count = int(cnt)
                            gini_row = compute_race_gini_from_win_rows(
                                race_id,
                                win_rows,
                                runner_count_hint=runner_count,
                            )
                            if gini_row is not None:
                                runner_count = int(gini_row["出走頭数"])
                                gini_value = float(gini_row["ジニ係数"])
                                gini_target_count = int(gini_row["ジニ対象頭数"])
                                gini_rows_all.append(
                                    {
                                        "race_id": str(race_id),
                                        "場所コード": "" if place_code is None else int(place_code),
                                        "馬場名": place_name,
                                        "出走頭数": int(gini_row["出走頭数"]),
                                        "ジニ対象頭数": int(gini_row["ジニ対象頭数"]),
                                        "ジニ係数": float(gini_row["ジニ係数"]),
                                    }
                                )
                        except Exception:
                            pass

                        # 払戻行（Sheets用）
                        for r in bundle.get("rows", []):
                            kind = str(r.get("券種", "")).strip()
                            if kind not in allowed_kinds:
                                continue
                            amount = parse_int_maybe(r.get("払戻金", 0))
                            if amount is None:
                                continue
                            payout_rows.append(
                                {
                                    "race_id": str(race_id),
                                    "券種": kind,
                                    "人気": str(r.get("人気", "")),
                                    "払戻金": int(amount),
                                    "買い目": str(r.get("買い目", "") or ""),
                                    "場所コード": place_code,
                                    "馬場名": place_name,
                                    "出走頭数": runner_count,
                                    "ジニ係数": gini_value,
                                    "ジニ対象頭数": gini_target_count,
                                }
                            )

                        if fetched_races >= int(nar_scan_total_limit):
                            stopped_reason = f"最大取得レース数（{nar_scan_total_limit}）に達したため停止しました。"
                            break
                    if stopped_reason:
                        break
                    if not rr_valid_any:
                        continue
                if stopped_reason:
                    break

            if stopped_reason:
                st.caption(stopped_reason)

            # 収支表示（買い目人気）
            if any(user_bets.get(k) for k in settlement_kinds) and settle_race_count > 0:
                st.subheader("収支（自分の買い目：馬券人気）")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                profit = int(settle_return_total) - int(settle_stake_total)
                roi = 0.0 if settle_stake_total <= 0 else float(settle_return_total) / float(settle_stake_total)
                col_s1.metric("対象レース数", str(int(settle_race_count)))
                col_s2.metric("掛け金合計(円)", f"{int(settle_stake_total):,}")
                col_s3.metric("払戻合計(円)", f"{int(settle_return_total):,}")
                col_s4.metric("利益(円)", f"{int(profit):,}")
                st.caption(f"回収率: {roi:.3f}")

                by_kind_rows: List[Dict[str, Any]] = []
                for k in settlement_kinds:
                    stake_k = int(settle_stake_by_kind.get(k, 0))
                    ret_k = int(settle_return_by_kind.get(k, 0))
                    by_kind_rows.append(
                        {
                            "馬券": k,
                            "購入人気": user_bets.get(k, []) or [],
                            "掛け金(円)": stake_k,
                            "払戻(円/100円)": ret_k,
                            "利益(円)": int(ret_k - stake_k),
                            "的中内訳": " / ".join(settle_hit_details.get(k, []) or []),
                        }
                    )
                st.dataframe(by_kind_rows, use_container_width=True)

            if gini_rows_all:
                st.subheader("レース別ジニ係数（単勝オッズ上位半分）")
                st.dataframe(gini_rows_all, use_container_width=True)

            if not payout_rows:
                status.update(label="完了（払戻0件）", state="complete")
                st.warning("払い戻しが見つかりませんでした（未確定/返還/ページ構造変更の可能性）。")
                st.stop()

            place_pairs = sorted(
                {
                    (str(r.get("場所コード", "") or ""), str(r.get("馬場名", "") or ""))
                    for r in payout_rows
                }
            )
            if place_pairs:
                st.caption(
                    "馬場ID一覧: "
                    + " / ".join([f"{code}:{name or '-'}" for code, name in place_pairs if code])
                )

            st.subheader("払戻（地方スキャン結果）")
            render_copy_button("払戻表をコピー(TSV)", rows_to_tsv(payout_rows), key="payout_scan_nar_table")
            st.dataframe(payout_rows, use_container_width=True)

            # このモードは Sheets へ追記する（認証情報が揃っている場合）
            can_write = sheets_enabled and has_service_account_source(
                sheets_sa_file, str(sheets_sa_path).strip()
            ) and bool(sheets_spreadsheet_id.strip())
            if not sheets_enabled:
                st.caption("Sheets記録はOFFです。")
            elif not can_write:
                st.info("Sheetsに書くには Spreadsheet ID とサービスアカウントJSONを指定してください。")
            else:
                status.update(label="処理中…（Sheetsへ書き込み）", state="running")
                sa_info = load_service_account_info(
                    uploaded=sheets_sa_file, local_path=str(sheets_sa_path).strip()
                )
                sheet_rows_by_name: Dict[str, List[List[Any]]] = {}
                gini_sheet_name = "ジニ係数"
                gini_by_race: Dict[str, Dict[str, Any]] = {
                    str(gr.get("race_id", "") or ""): gr for gr in gini_rows_all
                }
                for r in payout_rows:
                    kind = str(r.get("券種", "")).strip()
                    sheet_name = sheet_by_kind.get(kind)
                    if not sheet_name:
                        continue
                    race_id = str(r.get("race_id", "") or "")
                    gini_src = gini_by_race.get(race_id, {})
                    pop = parse_int_maybe(r.get("人気", ""))
                    amount = parse_int_maybe(r.get("払戻金", "")) or 0
                    place_code = r.get("場所コード")
                    gini_value = r.get("ジニ係数")
                    gini_float = parse_float_maybe(gini_value)
                    if gini_float is None:
                        gini_float = parse_float_maybe(gini_src.get("ジニ係数", ""))
                    runner_count = parse_int_maybe(r.get("出走頭数", ""))
                    if runner_count is None:
                        runner_count = parse_int_maybe(gini_src.get("出走頭数", ""))
                    sheet_rows_by_name.setdefault(sheet_name, []).append(
                        [
                            "" if pop is None else int(pop),
                            int(amount),
                            "" if place_code is None else int(place_code),
                            "" if gini_float is None else float(gini_float),
                            "" if runner_count is None else int(runner_count),
                        ]
                    )

                gini_sheet_rows: List[List[Any]] = []
                for gr in gini_rows_all:
                    place_code = parse_int_maybe(gr.get("場所コード", ""))
                    runner_count = parse_int_maybe(gr.get("出走頭数", ""))
                    top_count = parse_int_maybe(gr.get("ジニ対象頭数", ""))
                    gini_float = parse_float_maybe(gr.get("ジニ係数", ""))
                    gini_sheet_rows.append(
                        [
                            str(gr.get("race_id", "") or ""),
                            "" if place_code is None else int(place_code),
                            str(gr.get("馬場名", "") or ""),
                            "" if runner_count is None else int(runner_count),
                            "" if top_count is None else int(top_count),
                            "" if gini_float is None else float(gini_float),
                        ]
                    )

                total_written = 0
                for sheet_name, sheet_rows in sheet_rows_by_name.items():
                    google_sheets.ensure_sheet_exists(
                        spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                        sheet_name=sheet_name,
                        service_account_info=sa_info,
                    )
                    status.update(
                        label=f"処理中…（Sheets書き込み） {sheet_name} ({len(sheet_rows)}行)",
                        state="running",
                    )
                    google_sheets.append_rows(
                        spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                        sheet_name=sheet_name,
                        rows=sheet_rows,
                        service_account_info=sa_info,
                    )
                    total_written += len(sheet_rows)

                if gini_sheet_rows:
                    google_sheets.ensure_sheet_exists(
                        spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                        sheet_name=gini_sheet_name,
                        service_account_info=sa_info,
                    )
                    status.update(
                        label=f"処理中…（Sheets書き込み） {gini_sheet_name} ({len(gini_sheet_rows)}行)",
                        state="running",
                    )
                    google_sheets.append_rows(
                        spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                        sheet_name=gini_sheet_name,
                        rows=gini_sheet_rows,
                        service_account_info=sa_info,
                    )
                    total_written += len(gini_sheet_rows)
                st.success(f"スプレッドシートに記録しました（合計 {total_written} 行）。")

            status.update(label="完了", state="complete")
            st.session_state["_run_output"] = False
            st.stop()

        if mode_run == "払戻→Sheets（IDスキャン）":
            pp = re.sub(r"[^0-9]", "", scan_pp).zfill(2)[:2]
            if not (scan_year and pp and len(pp) == 2):
                raise ValueError("年(YYYY) と 馬場ID(PP 2桁) を指定してください。")

            allowed_kinds = {"単勝", "枠連", "馬単", "馬連", "三連複"}
            sheet_by_kind = {
                "単勝": "単勝大好き",
                "枠連": "枠連大好き",
                "馬単": "馬単大好き",
                "馬連": "馬連大好き",
                "三連複": "三連複大好き",
            }

            def mk_race_id(year: int, pp2: str, cc: int, dd: int, rr: int) -> str:
                return f"{year:04d}{pp2}{cc:02d}{dd:02d}{rr:02d}"

            def result_url_for(race_id: str) -> str:
                # 互換のため残す（現在は候補URLを順に試す）
                return f"https://race.netkeiba.com/race/result.html?race_id={race_id}&rf=race_submenu"

            def result_url_candidates_for(race_id: str) -> List[str]:
                # race_id 体系の判定が難しいため、JRA/NAR両方を試す
                return [
                    f"https://race.netkeiba.com/race/result.html?race_id={race_id}&rf=race_submenu",
                    f"https://nar.netkeiba.com/race/result.html?race_id={race_id}",
                ]

            payout_rows: List[Dict[str, Any]] = []
            gini_rows_all: List[Dict[str, Any]] = []
            fetched_races = 0
            stopped_reason = ""
            first_win_error: str = ""

            status = st.status("処理中…（IDスキャン）", expanded=False)
            progress = st.progress(0.0)

            # --- 自分の買い目（馬券人気）での収支集計 ---
            settlement_kinds = ("単勝", "枠連", "馬単", "馬連", "三連複")
            user_bets = st.session_state.get("user_bets", {}) or {}
            settle_stake_total = 0
            settle_return_total = 0
            settle_stake_by_kind: Dict[str, int] = {k: 0 for k in settlement_kinds}
            settle_return_by_kind: Dict[str, int] = {k: 0 for k in settlement_kinds}
            settle_hit_details: Dict[str, List[str]] = {k: [] for k in settlement_kinds}
            settle_race_count = 0

            # ルール:
            # - CC,DD を固定して RR=01..max を動かす
            # - 無効なRRで DD を進める（RR=01が無効なら DD が無効 → CC を進める）
            for cc in range(scan_cc_start, scan_cc_max + 1):
                dd_valid_any = False
                for dd in range(scan_dd_start, scan_dd_max + 1):
                    rr_valid_any = False
                    for rr in range(1, max(1, scan_rr_max) + 1):
                        race_id = mk_race_id(scan_year, pp, cc, dd, rr)

                        # まず結果ページで「有効レース（払戻あり）」を判定する
                        html = ""
                        bundle: Any = {}
                        selected_result_url = ""
                        found_result = False
                        status.update(label=f"処理中…（結果） race_id={race_id}", state="running")
                        for result_url in result_url_candidates_for(race_id):
                            try:
                                html_try = fetch_html(result_url)
                                if looks_like_invalid_result_page(html_try):
                                    continue
                                bundle_try = parse_payouts_bundle_from_result_html(html_try)
                                if result_bundle_has_any_payout(bundle_try, html_try):
                                    html = html_try
                                    bundle = bundle_try
                                    selected_result_url = result_url
                                    found_result = True
                                    break
                            except Exception:
                                continue
                        if not found_result:
                            if rr == 1:
                                # このDDは存在しない
                                break
                            # このDDのRRが尽きた/未確定
                            break

                        rr_valid_any = True
                        dd_valid_any = True
                        fetched_races += 1
                        if scan_total_limit > 0:
                            progress.progress(min(1.0, float(fetched_races) / float(scan_total_limit)))

                        place_code = parse_place_code_from_race_id(race_id)
                        race_ctx = parse_race_context_from_html(selected_result_url, html)
                        place_name = str(race_ctx.get("venue", "") or "") or place_name_from_code_any(place_code)
                        runner_count = parse_runner_count_from_result_html(html)
                        gini_row: Optional[Dict[str, Any]] = None
                        runner_count_from_odds: Optional[int] = None
                        # ジニ係数は任意（取れなくても払戻処理は継続）
                        try:
                            status.update(label=f"処理中…（オッズ） race_id={race_id}", state="running")
                            win_rows = fetch_win_rows_for_race(race_id)
                            if not win_rows:
                                win_rows = parse_win_odds_rows_from_result_html(html)
                            runner_count_from_odds = count_valid_win_rows(win_rows)
                            if runner_count is None and runner_count_from_odds is not None:
                                runner_count = int(runner_count_from_odds)
                            gini_row = compute_race_gini_from_win_rows(
                                race_id,
                                win_rows,
                                runner_count_hint=runner_count,
                            )
                        except Exception as e:
                            err = f"{type(e).__name__}: {e}"
                            if not first_win_error:
                                first_win_error = err

                        if gini_row is not None:
                            gini_rows_all.append(
                                {
                                    "race_id": str(race_id),
                                    "場所コード": "" if place_code is None else int(place_code),
                                    "馬場名": place_name,
                                    "出走頭数": int(gini_row["出走頭数"]),
                                    "ジニ対象頭数": int(gini_row["ジニ対象頭数"]),
                                    "ジニ係数": float(gini_row["ジニ係数"]),
                                }
                            )
                        gini_value = float(gini_row["ジニ係数"]) if gini_row is not None else None
                        gini_target_count = int(gini_row["ジニ対象頭数"]) if gini_row is not None else None

                        # 収支（買い目人気）をレース単位で集計
                        if any(user_bets.get(k) for k in settlement_kinds):
                            settle_race_count += 1
                            per_race = compute_settlement_from_payout_rows(
                                payout_rows=bundle.get("rows", []) if isinstance(bundle, dict) else [],
                                user_bets=user_bets,
                                kinds=settlement_kinds,
                                stake_per_bet_yen=100,
                            )
                            settle_stake_total += int(per_race["stake_total"])
                            settle_return_total += int(per_race["returned_total"])
                            for k in settlement_kinds:
                                settle_stake_by_kind[k] += int(per_race["stake_by_kind"].get(k, 0))
                                settle_return_by_kind[k] += int(per_race["returned_by_kind"].get(k, 0))
                                for d in per_race["hit_details"].get(k, []):
                                    settle_hit_details[k].append(f"{race_id} {d}")

                        for r in bundle.get("rows", []):
                            kind = str(r.get("券種", "")).strip()
                            if kind not in allowed_kinds:
                                continue
                            amount = parse_int_maybe(r.get("払戻金", 0))
                            if amount is None:
                                continue
                            payout_rows.append(
                                {
                                    "券種": kind,
                                    "人気": str(r.get("人気", "")),
                                    "金額": int(amount),
                                    "場所コード": place_code,
                                    "馬場名": place_name,
                                    "出走頭数": runner_count,
                                    "ジニ係数": gini_value,
                                    "ジニ対象頭数": gini_target_count,
                                }
                            )

                        if fetched_races >= scan_total_limit:
                            stopped_reason = (
                                f"最大取得レース数（{scan_total_limit}）に達したため停止しました。"
                            )
                            break
                    if stopped_reason:
                        break
                    if not rr_valid_any:
                        break
                if stopped_reason:
                    break
                if not dd_valid_any and cc == scan_cc_start:
                    stopped_reason = "指定範囲で有効なレースが見つかりませんでした。"
                    break

            if stopped_reason:
                st.caption(stopped_reason)

            # 収支表示（買い目人気）
            if any(user_bets.get(k) for k in settlement_kinds) and settle_race_count > 0:
                st.subheader("収支（自分の買い目：馬券人気）")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                profit = int(settle_return_total) - int(settle_stake_total)
                roi = 0.0 if settle_stake_total <= 0 else float(settle_return_total) / float(settle_stake_total)
                col_s1.metric("対象レース数", str(int(settle_race_count)))
                col_s2.metric("掛け金合計(円)", f"{int(settle_stake_total):,}")
                col_s3.metric("払戻合計(円)", f"{int(settle_return_total):,}")
                col_s4.metric("利益(円)", f"{int(profit):,}")
                st.caption(f"回収率: {roi:.3f}")

                by_kind_rows: List[Dict[str, Any]] = []
                for k in settlement_kinds:
                    stake_k = int(settle_stake_by_kind.get(k, 0))
                    ret_k = int(settle_return_by_kind.get(k, 0))
                    by_kind_rows.append(
                        {
                            "馬券": k,
                            "購入人気": user_bets.get(k, []) or [],
                            "掛け金(円)": stake_k,
                            "払戻(円/100円)": ret_k,
                            "利益(円)": int(ret_k - stake_k),
                            "的中内訳": " / ".join(settle_hit_details.get(k, []) or []),
                        }
                    )
                st.dataframe(by_kind_rows, use_container_width=True)

            if gini_rows_all:
                st.subheader("レース別ジニ係数（単勝オッズ上位半分）")
                st.dataframe(gini_rows_all, use_container_width=True)

            if not gini_rows_all and first_win_error:
                st.warning("単勝オッズが取得できないため、ジニ係数は未計算です。最初のエラー: " + first_win_error)

            if not payout_rows:
                st.warning("払戻が見つかりませんでした（未確定/返還/ページ構造変更の可能性）。ジニ係数のみ表示しています。")
            else:
                place_pairs = sorted(
                    {
                        (str(r.get("場所コード", "") or ""), str(r.get("馬場名", "") or ""))
                        for r in payout_rows
                    }
                )
                if place_pairs:
                    st.caption(
                        "馬場ID一覧: "
                        + " / ".join([f"{code}:{name or '-'}" for code, name in place_pairs if code])
                    )
                st.subheader("払戻（スキャン結果）")
                render_copy_button("払戻表をコピー(TSV)", rows_to_tsv(payout_rows), key="payout_scan_table")
                st.dataframe(payout_rows, use_container_width=True)

            # このモードは Sheets へ追記する（認証情報が揃っている場合）
            can_write = sheets_enabled and has_service_account_source(
                sheets_sa_file, str(sheets_sa_path).strip()
            ) and bool(sheets_spreadsheet_id.strip())
            if not sheets_enabled:
                st.caption("Sheets記録はOFFです。")
            elif not can_write:
                st.info("Sheetsに書くには Spreadsheet ID とサービスアカウントJSONを指定してください。")
            else:
                if not payout_rows and not gini_rows_all:
                    st.info("払戻とジニ係数の両方が0件のため、Sheetsへは書き込みません。")
                    status.update(label="完了", state="complete")
                    st.session_state["_run_output"] = False
                    st.stop()

                status.update(label="処理中…（Sheetsへ書き込み）", state="running")
                sa_info = load_service_account_info(uploaded=sheets_sa_file, local_path=str(sheets_sa_path).strip())
                sheet_rows_by_name: Dict[str, List[List[Any]]] = {}
                gini_sheet_name = "ジニ係数"
                for r in payout_rows:
                    kind = str(r.get("券種", "")).strip()
                    sheet_name = sheet_by_kind.get(kind)
                    if not sheet_name:
                        continue
                    pop = parse_int_maybe(r.get("人気", ""))
                    amount = parse_int_maybe(r.get("金額", "")) or 0
                    place_code = r.get("場所コード")
                    gini_value = r.get("ジニ係数")
                    gini_float = parse_float_maybe(gini_value)
                    runner_count = parse_int_maybe(r.get("出走頭数", ""))
                    sheet_rows_by_name.setdefault(sheet_name, []).append(
                        [
                            "" if pop is None else int(pop),
                            int(amount),
                            "" if place_code is None else int(place_code),
                            "" if gini_float is None else float(gini_float),
                            "" if runner_count is None else int(runner_count),
                        ]
                    )

                gini_sheet_rows: List[List[Any]] = []
                for gr in gini_rows_all:
                    place_code = parse_int_maybe(gr.get("場所コード", ""))
                    runner_count = parse_int_maybe(gr.get("出走頭数", ""))
                    top_count = parse_int_maybe(gr.get("ジニ対象頭数", ""))
                    gini_float = parse_float_maybe(gr.get("ジニ係数", ""))
                    gini_sheet_rows.append(
                        [
                            str(gr.get("race_id", "") or ""),
                            "" if place_code is None else int(place_code),
                            str(gr.get("馬場名", "") or ""),
                            "" if runner_count is None else int(runner_count),
                            "" if top_count is None else int(top_count),
                            "" if gini_float is None else float(gini_float),
                        ]
                    )

                total_written = 0
                for sheet_name, sheet_rows in sheet_rows_by_name.items():
                    google_sheets.ensure_sheet_exists(
                        spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                        sheet_name=sheet_name,
                        service_account_info=sa_info,
                    )
                    status.update(
                        label=f"処理中…（Sheets書き込み） {sheet_name} ({len(sheet_rows)}行)",
                        state="running",
                    )
                    google_sheets.append_rows(
                        spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                        sheet_name=sheet_name,
                        rows=sheet_rows,
                        service_account_info=sa_info,
                    )
                    total_written += len(sheet_rows)

                if gini_sheet_rows:
                    google_sheets.ensure_sheet_exists(
                        spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                        sheet_name=gini_sheet_name,
                        service_account_info=sa_info,
                    )
                    status.update(
                        label=f"処理中…（Sheets書き込み） {gini_sheet_name} ({len(gini_sheet_rows)}行)",
                        state="running",
                    )
                    google_sheets.append_rows(
                        spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                        sheet_name=gini_sheet_name,
                        rows=gini_sheet_rows,
                        service_account_info=sa_info,
                    )
                    total_written += len(gini_sheet_rows)

                st.success(f"スプレッドシートに記録しました（合計 {total_written} 行）。")

            status.update(label="完了", state="complete")
            st.session_state["_run_output"] = False
            st.stop()

        if mode_run == "払戻→Sheets（結果URLのみ）":
            if not payout_url.strip():
                raise ValueError("払い戻し（結果）URLを入力してください。")

            allowed_kinds = {"単勝", "枠連", "馬単", "馬連", "三連複"}
            sheet_by_kind = {
                "単勝": "単勝大好き",
                "枠連": "枠連大好き",
                "馬単": "馬単大好き",
                "馬連": "馬連大好き",
                "三連複": "三連複大好き",
            }

            def _extract_race_id(u: str) -> str:
                try:
                    qs = parse_qs(urlparse(u).query)
                    return str(qs.get("race_id", [""])[0] or "")
                except Exception:
                    return ""

            base_race_id = _extract_race_id(payout_url.strip())
            if payout_multi and (not base_race_id or len(base_race_id) < 2):
                raise ValueError("race_id がURLから取得できませんでした。race_id を含むURLを入力してください。")

            payout_rows: List[Dict[str, Any]] = []
            gini_rows_all: List[Dict[str, Any]] = []
            fetched_any = False

            status = st.status("処理中…（結果URL）", expanded=False)
            progress = st.progress(0.0)

            # --- 自分の買い目（馬券人気）での収支集計 ---
            settlement_kinds = ("単勝", "枠連", "馬単", "馬連", "三連複")
            user_bets = st.session_state.get("user_bets", {}) or {}
            settle_stake_total = 0
            settle_return_total = 0
            settle_stake_by_kind: Dict[str, int] = {k: 0 for k in settlement_kinds}
            settle_return_by_kind: Dict[str, int] = {k: 0 for k in settlement_kinds}
            settle_hit_details: Dict[str, List[str]] = {k: [] for k in settlement_kinds}
            settle_race_count = 0

            race_ids_to_fetch: List[str] = []
            if payout_multi:
                prefix = base_race_id[:-2]
                for rr in range(1, max(1, payout_max_round) + 1):
                    race_ids_to_fetch.append(prefix + f"{rr:02d}")
            else:
                race_ids_to_fetch.append(base_race_id)

            def _build_result_url(base_url: str, race_id: str) -> str:
                u = urlparse(base_url)
                qs = parse_qs(u.query)
                qs["race_id"] = [race_id]
                query_parts = []
                for k, vs in qs.items():
                    for v in vs:
                        query_parts.append(f"{k}={v}")
                query = "&".join(query_parts)
                path = u.path or "/race/result.html"
                netloc = u.netloc or "race.netkeiba.com"
                scheme = u.scheme or "https"
                return f"{scheme}://{netloc}{path}?{query}"

            for i, race_id in enumerate(race_ids_to_fetch, start=1):
                result_url = payout_url.strip()
                if payout_multi:
                    result_url = _build_result_url(payout_url.strip(), race_id)

                try:
                    status.update(label=f"処理中…（取得） race_id={race_id}", state="running")
                    payout_html = fetch_html(result_url)
                    if looks_like_invalid_result_page(payout_html):
                        raise RuntimeError(f"Invalid/empty result page: {result_url}")
                    bundle = parse_payouts_bundle_from_result_html(payout_html)
                except Exception:
                    if payout_multi:
                        break
                    raise

                # 「払戻がある=実施済みレース」だけを対象にする（返還/未確定/取消などは無視）
                if not result_bundle_has_any_payout(bundle, payout_html):
                    if payout_multi:
                        break
                    raise RuntimeError("払い戻しが見つかりませんでした（未確定/返還/ページ構造変更の可能性）。")

                fetched_any = True
                if payout_multi and payout_max_round > 0:
                    progress.progress(min(1.0, float(i) / float(payout_max_round)))

                place_code = parse_place_code_from_race_id(race_id)
                race_ctx = parse_race_context_from_html(result_url, payout_html)
                place_name = str(race_ctx.get("venue", "") or "") or place_name_from_code_any(place_code)
                runner_count = parse_runner_count_from_result_html(payout_html)
                gini_value: Optional[float] = None
                gini_target_count: Optional[int] = None

                # 収支（買い目人気）をレース単位で集計
                if any(user_bets.get(k) for k in settlement_kinds):
                    settle_race_count += 1
                    per_race = compute_settlement_from_payout_rows(
                        payout_rows=bundle.get("rows", []) if isinstance(bundle, dict) else [],
                        user_bets=user_bets,
                        kinds=settlement_kinds,
                        stake_per_bet_yen=100,
                    )
                    settle_stake_total += int(per_race["stake_total"])
                    settle_return_total += int(per_race["returned_total"])
                    for k in settlement_kinds:
                        settle_stake_by_kind[k] += int(per_race["stake_by_kind"].get(k, 0))
                        settle_return_by_kind[k] += int(per_race["returned_by_kind"].get(k, 0))
                        for d in per_race["hit_details"].get(k, []):
                            settle_hit_details[k].append(f"{race_id} {d}")

                # 単勝オッズを取得して、上位半分のジニ係数を計算
                try:
                    status.update(label=f"処理中…（オッズ） race_id={race_id}", state="running")
                    win_rows = fetch_win_rows_for_race(race_id)
                    if not win_rows:
                        win_rows = parse_win_odds_rows_from_result_html(payout_html)
                    cnt = count_valid_win_rows(win_rows)
                    if runner_count is None and cnt is not None:
                        runner_count = int(cnt)
                    gini_row = compute_race_gini_from_win_rows(
                        race_id,
                        win_rows,
                        runner_count_hint=runner_count,
                    )
                    if gini_row is not None:
                        gini_value = float(gini_row["ジニ係数"])
                        gini_target_count = int(gini_row["ジニ対象頭数"])
                        gini_rows_all.append(
                            {
                                "race_id": str(race_id),
                                "場所コード": "" if place_code is None else int(place_code),
                                "馬場名": place_name,
                                "出走頭数": int(gini_row["出走頭数"]),
                                "ジニ対象頭数": int(gini_row["ジニ対象頭数"]),
                                "ジニ係数": float(gini_row["ジニ係数"]),
                            }
                        )
                except Exception:
                    pass

                # 払戻行（Sheets用）
                for r in bundle.get("rows", []):
                    kind = str(r.get("券種", "")).strip()
                    if kind not in allowed_kinds:
                        continue
                    amount = parse_int_maybe(r.get("払戻金", 0))
                    if amount is None:
                        continue
                    payout_rows.append(
                        {
                            "券種": kind,
                            "人気": str(r.get("人気", "")),
                            "金額": int(amount),
                            "場所コード": place_code,
                            "馬場名": place_name,
                            "出走頭数": runner_count,
                            "ジニ係数": gini_value,
                            "ジニ対象頭数": gini_target_count,
                        }
                    )

            if not fetched_any:
                status.update(label="処理失敗（取得0件）", state="error")
                raise RuntimeError("払い戻しを取得できませんでした（URL/接続/ブロックの可能性）。")

            if not payout_rows:
                status.update(label="完了（払戻0件）", state="complete")
                st.warning("払い戻しが見つかりませんでした（未確定/返還/ページ構造変更の可能性）。")
                st.stop()

            # 収支表示（買い目人気）
            if any(user_bets.get(k) for k in settlement_kinds) and settle_race_count > 0:
                st.subheader("収支（自分の買い目：馬券人気）")
                col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                profit = int(settle_return_total) - int(settle_stake_total)
                roi = 0.0 if settle_stake_total <= 0 else float(settle_return_total) / float(settle_stake_total)
                col_s1.metric("対象レース数", str(int(settle_race_count)))
                col_s2.metric("掛け金合計(円)", f"{int(settle_stake_total):,}")
                col_s3.metric("払戻合計(円)", f"{int(settle_return_total):,}")
                col_s4.metric("利益(円)", f"{int(profit):,}")
                st.caption(f"回収率: {roi:.3f}")

                by_kind_rows: List[Dict[str, Any]] = []
                for k in settlement_kinds:
                    stake_k = int(settle_stake_by_kind.get(k, 0))
                    ret_k = int(settle_return_by_kind.get(k, 0))
                    by_kind_rows.append(
                        {
                            "馬券": k,
                            "購入人気": user_bets.get(k, []) or [],
                            "掛け金(円)": stake_k,
                            "払戻(円/100円)": ret_k,
                            "利益(円)": int(ret_k - stake_k),
                            "的中内訳": " / ".join(settle_hit_details.get(k, []) or []),
                        }
                    )
                st.dataframe(by_kind_rows, use_container_width=True)

            if gini_rows_all:
                st.subheader("レース別ジニ係数（単勝オッズ上位半分）")
                st.dataframe(gini_rows_all, use_container_width=True)

            st.subheader("払戻（結果URL）")
            render_copy_button("払戻表をコピー(TSV)", rows_to_tsv(payout_rows), key="payout_only_table")
            st.dataframe(payout_rows, use_container_width=True)

            # このモードは Sheets へ追記する（認証情報が揃っている場合）
            can_write = sheets_enabled and has_service_account_source(
                sheets_sa_file, str(sheets_sa_path).strip()
            ) and bool(sheets_spreadsheet_id.strip())
            if not sheets_enabled:
                st.caption("Sheets記録はOFFです。")
            elif not can_write:
                st.info("Sheetsに書くには Spreadsheet ID とサービスアカウントJSONを指定してください。")
            else:
                try:
                    sa_info = load_service_account_info(
                        uploaded=sheets_sa_file, local_path=str(sheets_sa_path).strip()
                    )
                    sheet_rows_by_name: Dict[str, List[List[Any]]] = {}
                    gini_sheet_name = "ジニ係数"
                    for r in payout_rows:
                        kind = str(r.get("券種", "")).strip()
                        sheet_name = sheet_by_kind.get(kind)
                        if not sheet_name:
                            continue
                        pop = parse_int_maybe(r.get("人気", ""))
                        amount = (
                            parse_int_maybe(r.get("金額", ""))
                            or parse_int_maybe(r.get("払戻金", ""))
                            or 0
                        )
                        place_code = r.get("場所コード")
                        gini_value = r.get("ジニ係数")
                        gini_float = parse_float_maybe(gini_value)
                        runner_count = parse_int_maybe(r.get("出走頭数", ""))
                        sheet_rows_by_name.setdefault(sheet_name, []).append(
                            [
                                "" if pop is None else int(pop),
                                int(amount),
                                "" if place_code is None else int(place_code),
                                "" if gini_float is None else float(gini_float),
                                "" if runner_count is None else int(runner_count),
                            ]
                        )

                    gini_sheet_rows: List[List[Any]] = []
                    for gr in gini_rows_all:
                        place_code = parse_int_maybe(gr.get("場所コード", ""))
                        runner_count = parse_int_maybe(gr.get("出走頭数", ""))
                        top_count = parse_int_maybe(gr.get("ジニ対象頭数", ""))
                        gini_float = parse_float_maybe(gr.get("ジニ係数", ""))
                        gini_sheet_rows.append(
                            [
                                str(gr.get("race_id", "") or ""),
                                "" if place_code is None else int(place_code),
                                str(gr.get("馬場名", "") or ""),
                                "" if runner_count is None else int(runner_count),
                                "" if top_count is None else int(top_count),
                                "" if gini_float is None else float(gini_float),
                            ]
                        )

                    total_written = 0
                    for sheet_name, sheet_rows in sheet_rows_by_name.items():
                        google_sheets.ensure_sheet_exists(
                            spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                            sheet_name=sheet_name,
                            service_account_info=sa_info,
                        )
                        status.update(
                            label=f"処理中…（Sheets書き込み） {sheet_name} ({len(sheet_rows)}行)",
                            state="running",
                        )
                        google_sheets.append_rows(
                            spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                            sheet_name=sheet_name,
                            rows=sheet_rows,
                            service_account_info=sa_info,
                        )
                        total_written += len(sheet_rows)

                    if gini_sheet_rows:
                        google_sheets.ensure_sheet_exists(
                            spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                            sheet_name=gini_sheet_name,
                            service_account_info=sa_info,
                        )
                        status.update(
                            label=f"処理中…（Sheets書き込み） {gini_sheet_name} ({len(gini_sheet_rows)}行)",
                            state="running",
                        )
                        google_sheets.append_rows(
                            spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                            sheet_name=gini_sheet_name,
                            rows=gini_sheet_rows,
                            service_account_info=sa_info,
                        )
                        total_written += len(gini_sheet_rows)

                    st.success(f"スプレッドシートに記録しました（合計 {total_written} 行）。")
                except google_sheets.GoogleSheetsUnavailable as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"書き込みに失敗しました: {e}")

            status.update(label="完了", state="complete")
            st.session_state["_run_output"] = False
            st.stop()

        # --- 予想（オッズ＋結果で計算） ---
        with st.spinner("取得中…"):
            html = fetch_html(odds_url_run)
            odds_json = build_odds_json(odds_url_run, html)
            odds_json = filter_odds_json_to_top6(odds_json)
            st.session_state["odds_json"] = odds_json
            if save_odds_local:
                save_json_file(Path("odds.json"), odds_json)

        st.success("取得できました")

        race_ctx = parse_race_context_from_html(odds_url_run, html)
        st.session_state["race_ctx"] = race_ctx
        race_parts: List[str] = []
        if race_ctx.get("date"):
            race_parts.append(str(race_ctx["date"]))
        if race_ctx.get("venue"):
            race_parts.append(str(race_ctx["venue"]))
        if race_ctx.get("race_no") is not None:
            race_parts.append(f'{int(race_ctx["race_no"])}R')
        if race_ctx.get("race_id"):
            race_parts.append(f'(race_id={race_ctx["race_id"]})')
        if race_parts:
            st.caption("レース情報: " + " ".join(race_parts))

        st.caption(f"取得時刻(UTC): {odds_json['meta']['fetched_at']}")

        tables = compute_main_tables(odds_json)

        # 枠連のEVは「払戻→Sheets（スキャン）」側で race_id から同時取得する仕様に変更したため、
        # 予想モードではここでは表示しない。

        # まず「買う買い目」だけを最上部に出す（見たい情報を最短で見せる）
        # 参考表示用：三連複のEV上位6件
        trio_positive = sorted(
            tables.get("sanrenpuku_ev", []),
            key=lambda r: float(r.get("EV", 0.0)),
            reverse=True,
        )[:6]
        if str(pick_mode).startswith("確定枠"):
            market_rows = list(tables.get("market", []))
            pop_sorted = sorted(
                market_rows,
                key=lambda r: int(r.get("人気", 10**9)),
            )
            pop_nums = [int(r.get("馬番", 0)) for r in pop_sorted if int(r.get("馬番", 0)) > 0]
            rank1 = pop_nums[0] if len(pop_nums) >= 1 else 0
            rank2 = pop_nums[1] if len(pop_nums) >= 2 else 0
            rank3 = pop_nums[2] if len(pop_nums) >= 3 else 0

            # テーブルから単勝の詳細を引く
            tansho_by_num = {int(r.get("馬番", 0)): r for r in tables.get("tansho_ev", [])}

            def make_tansho_rec(num: int) -> Dict[str, Any]:
                row = tansho_by_num.get(int(num), {})
                p = float(row.get("当選確率(%)", 0.0)) / 100.0
                evu = float(row.get("EV", 0.0))
                ks = float(row.get("Kelly(%)", 0.0)) / 100.0
                ll = float(row.get("p×log(o)", 0.0))
                return {
                    "種別": "単勝",
                    "買い目": str(int(num)),
                    "EVU": evu,
                    "P": p,
                    "KS": ks,
                    "LL": ll,
                    "LU": ll,
                    "LS": ll,
                    "_pass_and": True,
                    "_pass_p_only": False,
                }

            recs = []

            # 追加：ワイド（人気1位-3位）を買う
            if rank1 and rank3 and rank1 != rank3:
                a, b = sorted([int(rank1), int(rank3)])
                wide_bet = f"{a}-{b}"
                wide_row = next(
                    (r for r in tables.get("wide_ev", []) if str(r.get("買い目", "")) == wide_bet),
                    None,
                )
                if wide_row is not None:
                    p = float(wide_row.get("当選確率(%)", 0.0)) / 100.0
                    evu = float(wide_row.get("EV上限", 0.0))
                    ks = float(wide_row.get("Kelly上限(%)", 0.0)) / 100.0
                    ll = float(wide_row.get("p×log(o)上限", 0.0))
                else:
                    p = 0.0
                    evu = 0.0
                    ks = 0.0
                    ll = 0.0
                recs.append(
                    {
                        "種別": "ワイド",
                        "買い目": wide_bet,
                        "EVU": evu,
                        "P": p,
                        "KS": ks,
                        "LL": ll,
                        "LU": ll,
                        "LS": ll,
                        "_pass_and": True,
                        "_pass_p_only": False,
                    }
                )

            # 追加：三連複のEV上位5位も買う（各500円）
            top5_trios = sorted(
                tables.get("sanrenpuku_ev", []),
                key=lambda r: float(r.get("EV", 0.0)),
                reverse=True,
            )[:5]
            for r in top5_trios:
                bet = str(r.get("買い目", ""))
                if not bet:
                    continue
                recs.append(
                    {
                        "種別": "三連複",
                        "買い目": bet,
                        "EVU": float(r.get("EV", 0.0)),
                        "P": float(r.get("当選確率(%)", 0.0)) / 100.0,
                        "KS": float(r.get("Kelly(%)", 0.0)) / 100.0,
                        "LL": float(r.get("p×log(o)", 0.0)),
                        "LU": float(r.get("p×log(o)", 0.0)),
                        "LS": float(r.get("p×log(o)", 0.0)),
                        "_pass_and": True,
                        "_pass_p_only": False,
                    }
                )

            # 重複排除（同じ買い目が複数回入るのを防ぐ）
            uniq: List[Dict[str, Any]] = []
            seen_keys: set[Tuple[str, str]] = set()
            for r in recs:
                k = (str(r.get("種別", "")), str(r.get("買い目", "")))
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                uniq.append(r)
            recs = uniq
        else:
            recs, _ = build_recommendations(
                tables,
                ev_and_min=float(ev_and_threshold),
                prob_and_min=float(prob_threshold_percent) / 100.0,
                prob_or_min=float(prob_or_threshold_percent) / 100.0,
            )

        # 購入対象から除外（複勝は買わない / 単勝も買わない）
        recs = [r for r in recs if str(r.get("種別", "")) not in ("複勝", "単勝")]

        rec_rows: List[Dict[str, Any]] = []
        bet_fractions: List[float] = []
        for r in recs:
            ks = float(r["KS"])
            kind = str(r["種別"])
            bet_fraction = compute_bet_fraction_for_kind(kind, ks)
            bet_fractions.append(bet_fraction)
            rec_rows.append(
                {
                    "馬券": kind,
                    "買い目": r["買い目"],
                    "掛け金(円)": 0,  # 後で予算内に収まるよう配分
                    # 以下は詳細表示用（購入リストには出さない）
                    "_EVU": float(r["EVU"]),
                    "_P": float(r.get("P", 0.0)),
                    "_KS": ks,
                    "_BET_FRACTION": bet_fraction,
                    "_LS": float(r["LS"]),
                    "_LL": float(r["LL"]),
                    "_LU": float(r["LU"]),
                    "_PASS_AND": bool(r.get("_pass_and", False)),
                    "_PASS_P_ONLY": bool(r.get("_pass_p_only", False)),
                }
            )

        # 掛け金ルール（ユーザー指定）
        # - ワイド（人気1-3）: 2000円
        # - 三連複（EV上位5）: 各500円
        fixed_stake_by_kind: Dict[str, int] = {"三連複": 500, "ワイド": 2000}

        # まず固定券種に掛け金を付ける（予算は超えても固定額を優先）
        fixed_total = 0
        remaining_indices: List[int] = []
        remaining_fractions: List[float] = []
        for idx, (row, frac) in enumerate(zip(rec_rows, bet_fractions)):
            kind = str(row.get("馬券", ""))
            if kind in fixed_stake_by_kind:
                stake = int(fixed_stake_by_kind[kind])
                row["掛け金(円)"] = stake
                fixed_total += stake
            else:
                remaining_indices.append(idx)
                remaining_fractions.append(float(frac))

        # 固定以外の券種は、残り予算内で従来ロジックに従う
        remaining_budget_yen = max(0, int(budget_yen) - int(fixed_total))
        if remaining_indices:
            remaining_stakes = allocate_stakes_with_budget_cap(
                remaining_budget_yen,
                remaining_fractions,
                min_round_up_yen=int(min_round_up_yen),
                max_stake_per_bet_yen=0,
            )
            for idx, stake in zip(remaining_indices, remaining_stakes):
                rec_rows[idx]["掛け金(円)"] = int(stake)

        # 表示用：掛け割合は「実際に割り当てた掛け金」から再計算して整合させる
        usable_budget = (int(budget_yen) // 100) * 100
        for row in rec_rows:
            stake_yen = int(row.get("掛け金(円)", 0))
            row["_BET_FRACTION"] = 0.0 if usable_budget <= 0 else float(stake_yen) / float(usable_budget)

        # 予算超過チェック（固定額 + 三連複最低額の影響で超えることがある）
        total_now = sum(int(x.get("掛け金(円)", 0)) for x in rec_rows)
        if int(budget_yen) > 0 and total_now > int(budget_yen):
            st.warning(
                f"掛け金の合計（{total_now}円）が予算（{int(budget_yen)}円）を超えています。"
            )
        # 「賭け金0円を非表示」は購入リストだけに適用（条件表は条件を満たした全件を出す）
        buy_rows = rec_rows
        if hide_zero_stake:
            buy_rows = [x for x in rec_rows if int(x["掛け金(円)"]) > 0]

        # 購入リストは「同じ券種・同じ買い目（正規化後）」を合算して表示する
        buy_agg: Dict[Tuple[str, str], int] = {}
        for x in buy_rows:
            kind = str(x["馬券"])
            bet_key = normalize_bet_for_lookup(kind, str(x["買い目"]))
            buy_agg[(kind, bet_key)] = buy_agg.get((kind, bet_key), 0) + int(x["掛け金(円)"])
        buy_table: List[Dict[str, Any]] = [
            {"馬券": kind, "買い目": bet, "掛け金(円)": stake}
            for (kind, bet), stake in buy_agg.items()
        ]
        total_stake = sum(int(x["掛け金(円)"]) for x in buy_table)

        # 払い戻しが取得できる場合は、先に照合して「収支」を最上部に表示する
        settlement_available = False
        settlement_profit = 0
        settlement_roi = 0.0
        settlement_total_return = 0
        settlement_rows: List[Dict[str, Any]] = []
        payouts_json: Optional[Dict[str, Any]] = None

        if payout_url.strip() and buy_table:
            with st.spinner("払い戻し取得中…"):
                payout_html = fetch_html(payout_url.strip())
                payout_race_ctx = parse_race_context_from_html(payout_url.strip(), payout_html)
                st.session_state["payout_race_ctx"] = payout_race_ctx
                bundle = parse_payouts_bundle_from_result_html(payout_html)
                payouts_json = {
                    "meta": {
                        "source_url": payout_url.strip(),
                        "fetched_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "payouts": bundle["payouts"],
                    "rows": bundle["rows"],
                }
                st.session_state["payouts_json"] = payouts_json
                pr_parts: List[str] = []
                if payout_race_ctx.get("date"):
                    pr_parts.append(str(payout_race_ctx["date"]))
                if payout_race_ctx.get("venue"):
                    pr_parts.append(str(payout_race_ctx["venue"]))
                if payout_race_ctx.get("race_no") is not None:
                    pr_parts.append(f'{int(payout_race_ctx["race_no"])}R')
                if payout_race_ctx.get("race_id"):
                    pr_parts.append(f'(race_id={payout_race_ctx["race_id"]})')
                if pr_parts:
                    st.caption("結果レース情報: " + " ".join(pr_parts))
                if save_payouts_local:
                    save_json_file(Path("payouts.json"), payouts_json)
                    st.caption("保存しました: payouts.json")

            total_return = 0
            for row in buy_table:
                kind_ui = str(row["馬券"])
                kind = kind_ui
                bet = str(row["買い目"])
                stake = int(row["掛け金(円)"])
                bet_key = normalize_bet_for_lookup(kind, bet)
                per100 = payouts_json["payouts"].get(kind, {}).get(bet_key) if payouts_json else None
                returned = 0
                hit = per100 is not None and stake > 0
                if hit and per100 is not None:
                    returned = int(per100) * (stake // 100)
                total_return += returned
                settlement_rows.append(
                    {
                        "馬券": kind_ui,
                        "買い目": bet,
                        "照合キー": bet_key,
                        "掛け金(円)": stake,
                        "払戻(円/100円)": "" if per100 is None else int(per100),
                        "払戻合計(円)": returned,
                        "的中": "○" if hit and returned > 0 else "",
                    }
                )

            settlement_available = True
            settlement_total_return = int(total_return)
            settlement_profit = int(total_return) - int(total_stake)
            settlement_roi = 0.0 if total_stake <= 0 else float(total_return) / float(total_stake)

        if settlement_available:
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("合計掛け金(円)", f"{int(total_stake):,}")
            col_m2.metric("払戻合計(円)", f"{int(settlement_total_return):,}")
            col_m3.metric("収支(円)", f"{int(settlement_profit):,}")
            col_m4.metric("回収率", f"{settlement_roi:.3f}")

            # Google Sheets へ記録（任意）
            if sheets_enabled:
                race_ctx_for_report = st.session_state.get("payout_race_ctx") or st.session_state.get(
                    "race_ctx", {}
                )
                date_str = str(race_ctx_for_report.get("date", "") or "")
                date_md = date_md_jp_from_ymd(date_str)
                date_for_sheets = date_for_sheets_from_ymd(date_str)
                venue = str(race_ctx_for_report.get("venue", "") or "")
                race_no = race_ctx_for_report.get("race_no")
                race_no_str = "" if race_no is None else f"{int(race_no)}R"

                report_values = [
                    date_for_sheets,
                    venue,
                    race_no_str,
                    int(settlement_profit),
                    float(f"{settlement_roi:.6f}"),
                ]

                can_write = has_service_account_source(
                    sheets_sa_file, str(sheets_sa_path).strip()
                ) and bool(sheets_spreadsheet_id.strip())
                if not can_write:
                    if not sheets_spreadsheet_id.strip():
                        st.info("Sheetsに書くには Spreadsheet ID を入力してください。")
                    else:
                        st.info("Sheetsに書くには、サービスアカウントJSONを指定してください（ファイル名 or アップロード）。")
                else:
                    st.caption(
                        f"Sheets記録内容: 日付={date_md} / 場所={venue} / ラウンド={race_no_str} / 収支={int(settlement_profit)} / 回収率={settlement_roi:.3f}"
                    )

                    # 出力ボタン押下と同時に、自動で 1回だけ追記する
                    # ※同じレースでも何度でも記録できる（重複防止はしない）
                    try:
                        sa_info = load_service_account_info(
                            uploaded=sheets_sa_file, local_path=str(sheets_sa_path).strip()
                        )
                        google_sheets.append_row(
                            spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                            sheet_name=str(sheets_sheet_name).strip(),
                            values=report_values,
                            service_account_info=sa_info,
                        )
                        st.success("スプレッドシートに記録しました。")
                    except google_sheets.GoogleSheetsUnavailable as e:
                        st.error(str(e))
                        if st.button(
                            "Sheets連携ライブラリを自動インストールして再実行",
                            key="install_sheets_deps",
                        ):
                            try:
                                with st.spinner("pip install 実行中…"):
                                    install_google_sheets_deps()
                                with st.spinner("Sheetsへ書き込み再実行中…"):
                                    google_sheets.append_row(
                                        spreadsheet_id=str(sheets_spreadsheet_id).strip(),
                                        sheet_name=str(sheets_sheet_name).strip(),
                                        values=report_values,
                                        service_account_info=sa_info,
                                    )
                                st.success("インストール＋記録が完了しました。")
                            except Exception as ie:
                                st.error(f"インストールに失敗しました: {ie}")
                    except Exception as e:
                        st.error(f"書き込みに失敗しました: {e}")

        st.subheader("購入リスト（馬券 / 買い目 / 掛け金）")
        if rec_rows:
            render_copy_button("購入リストをコピー(TSV)", rows_to_tsv(buy_table), key="buy_table")
            st.dataframe(buy_table, use_container_width=True)
            usable_budget = (int(budget_yen) // 100) * 100
            st.caption(
                f"合計掛け金: {total_stake} 円 / 予算: {int(budget_yen)} 円（100円単位で使える予算: {usable_budget} 円）"
            )
            st.caption("※購入は確定枠のみ（ワイド:人気1-3を2000円 / 三連複:EV上位5を各500円）。")

            st.subheader("払い戻し集計（結果URLと照合）")
            if payout_url.strip() and payouts_json:
                # 取得できた券種の件数（デバッグ用。UIはうるさくしないため expander 内）
                with st.expander("デバッグ：払戻取得状況", expanded=False):
                    kind_counts = {
                        k: len(payouts_json.get("payouts", {}).get(k, {}) or {})
                        for k in ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"]
                    }
                    st.write("券種別 件数:", kind_counts)
                    if kind_counts.get("ワイド", 0) == 0:
                        st.warning(
                            "ワイドが0件です。結果ページの払戻表のHTML構造が想定と違う可能性があります。"
                        )

                st.download_button(
                    label="payouts.json をダウンロード",
                    data=json.dumps(payouts_json, ensure_ascii=False, indent=2),
                    file_name="payouts.json",
                    mime="application/json",
                )

                st.subheader("払戻金（結果URL）")
                payout_table = []
                for r in payouts_json.get("rows", []):
                    payout_table.append(
                        {
                            "券種": r["券種"],
                            "買い目": r["買い目"],
                            "払戻金": f'{int(r["払戻金"]):,}円',
                            "人気": r.get("人気", ""),
                        }
                    )
                render_copy_button(
                    "払戻金表をコピー(TSV)", rows_to_tsv(payout_table), key="payout_table"
                )
                st.dataframe(payout_table, use_container_width=True)

                st.subheader("照合結果（購入リスト × 払戻）")
                render_copy_button(
                    "照合結果をコピー(TSV)", rows_to_tsv(settlement_rows), key="settle_table"
                )
                # ワイドが本当に照合できているか分かるように、券種ごとの照合件数を表示
                by_kind_total: Dict[str, int] = {}
                by_kind_matched: Dict[str, int] = {}
                for row in buy_table:
                    k = str(row["馬券"])
                    by_kind_total[k] = by_kind_total.get(k, 0) + 1
                for row in settlement_rows:
                    k = str(row["馬券"])
                    matched = row["払戻(円/100円)"] != ""
                    if matched:
                        by_kind_matched[k] = by_kind_matched.get(k, 0) + 1
                wide_total = by_kind_total.get("ワイド", 0)
                wide_matched = by_kind_matched.get("ワイド", 0)
                trio_total = by_kind_total.get("三連複", 0)
                trio_matched = by_kind_matched.get("三連複", 0)
                st.caption(f"ワイド照合: {wide_matched}/{wide_total} / 三連複照合: {trio_matched}/{trio_total}（払戻が見つかった件数/購入件数）")
                st.dataframe(settlement_rows, use_container_width=True)
                st.caption(
                    f"合計掛け金: {total_stake} 円 / 払戻合計: {settlement_total_return} 円 / 収支: {settlement_profit} 円 / 回収率: {settlement_roi:.3f}"
                )
            else:
                st.info("払い戻しURLを入力してください。")
        else:
            st.info("条件を満たす買い目がありません（または掛け金が全て0円です）。")

        st.subheader("勝率")
        st.caption("市場確率は『人気上位6頭のみ』を対象に作成（7位以降はデータから除外）")
        render_copy_button("勝率表をコピー(TSV)", rows_to_tsv(tables["market"]), key="market_table")
        st.dataframe(tables["market"], use_container_width=True)

        # --- 三連複（EV上位6件）に基づく派生候補 ---
        trio_positive_rows = sorted(
            tables.get("sanrenpuku_ev", []),
            key=lambda r: float(r.get("EV", 0.0)),
            reverse=True,
        )[:6]
        if trio_positive_rows:
            st.subheader("三連複：EV上位6 の買い目")
            trio_table = [
                {
                    "買い目": str(r.get("買い目", "")),
                    "当選確率(%)": float(r.get("当選確率(%)", 0.0)),
                    "オッズ": float(r.get("オッズ", 0.0)),
                    "EV": float(r.get("EV", 0.0)),
                }
                for r in trio_positive_rows
            ]
            render_copy_button("三連複EV上位6をコピー(TSV)", rows_to_tsv(trio_table), key="trio_pos")
            st.dataframe(trio_table, use_container_width=True)

        # --- 三連複：全組み合わせ（上位6頭なら20通り） ---
        numbers = sorted({int(r.get("馬番", 0)) for r in tables.get("market", []) if int(r.get("馬番", 0)) > 0})
        if len(numbers) >= 3:
            prob_by_number = {
                int(r.get("馬番", 0)): float(r.get("市場確率(%)", 0.0)) / 100.0
                for r in tables.get("market", [])
                if int(r.get("馬番", 0)) > 0
            }
            odds_trio_map = {}
            try:
                odds_trio_map = dict((odds_json.get("tickets") or {}).get("trio") or {})
            except Exception:
                odds_trio_map = {}

            sanrenpuku_all_table: List[Dict[str, Any]] = []
            for a, b, c in itertools.combinations(numbers, 3):
                bet_key = f"{a}-{b}-{c}"
                p = float(ticket_probs.sanrenpuku_probability(prob_by_number, a, b, c))
                odds_val = odds_trio_map.get(bet_key)
                odds_f = None
                if odds_val is not None:
                    try:
                        odds_f = float(odds_val)
                    except Exception:
                        odds_f = None
                ev_val: Any = ""
                if odds_f and odds_f > 0:
                    ev_val = float(compute_ev(p, odds_f))
                payout_per100 = ""
                if payouts_json and isinstance(payouts_json.get("payouts"), dict):
                    payout_per100 = payouts_json["payouts"].get("三連複", {}).get(bet_key, "")
                sanrenpuku_all_table.append(
                    {
                        "買い目": bet_key,
                        "当選確率(%)": round(p * 100, 4),
                        "オッズ": "" if odds_f is None else odds_f,
                        "EV": ev_val,
                        "払戻(円/100円)": payout_per100,
                        "結果": "的中" if payout_per100 != "" else "",
                    }
                )
            sanrenpuku_all_table.sort(key=lambda r: float(r["EV"]) if r["EV"] != "" else -1e9, reverse=True)

            st.subheader(f"三連複：全組み合わせ（{len(sanrenpuku_all_table)}通り）")
            st.caption("※オッズは取得できた組み合わせのみ表示（空欄はオッズ未取得）。")
            render_copy_button(
                "三連複：全組み合わせをコピー(TSV)",
                rows_to_tsv(sanrenpuku_all_table),
                key="sanrenpuku_all",
            )
            st.dataframe(sanrenpuku_all_table, use_container_width=True)

            allowed_numbers: set[int] = set()
            for r in trio_positive_rows:
                nums = [int(x) for x in re.findall(r"\d+", str(r.get("買い目", "")))]
                allowed_numbers.update(nums)
            st.caption(f"派生候補に使う馬番（EV上位6三連複に含まれる馬）: {sorted(allowed_numbers)}")

            def row_numbers(row: Dict[str, Any]) -> List[int]:
                if "馬番" in row and row["馬番"] != "":
                    return [int(row["馬番"])]
                return [int(x) for x in re.findall(r"\d+", str(row.get("買い目", "")))]

            derived_rows: List[Dict[str, Any]] = []

            def add_single(
                kind: str, row: Dict[str, Any], odds_key: str, ev_key: str = "EV"
            ) -> None:
                nums = row_numbers(row)
                if nums and all(n in allowed_numbers for n in nums):
                    derived_rows.append(
                        {
                            "券種": kind,
                            "買い目": str(row.get("買い目", row.get("馬番", ""))),
                            "オッズ": float(row.get(odds_key, 0.0)),
                            "下限": "",
                            "上限": "",
                            "EV": float(row.get(ev_key, 0.0)),
                            "EV下限": "",
                            "EV上限": "",
                        }
                    )

            def add_range(
                kind: str,
                row: Dict[str, Any],
                lo_key: str,
                hi_key: str,
                ev_lo: str,
                ev_hi: str,
            ) -> None:
                nums = row_numbers(row)
                if nums and all(n in allowed_numbers for n in nums):
                    derived_rows.append(
                        {
                            "券種": kind,
                            "買い目": str(row.get("買い目", row.get("馬番", ""))),
                            "オッズ": "",
                            "下限": float(row.get(lo_key, 0.0)),
                            "上限": float(row.get(hi_key, 0.0)),
                            "EV": "",
                            "EV下限": float(row.get(ev_lo, 0.0)),
                            "EV上限": float(row.get(ev_hi, 0.0)),
                        }
                    )

            # 派生候補（参考表示）は「単勝/複勝/ワイド」だけを表示する
            for r in tables.get("tansho_ev", []):
                add_single(
                    "単勝",
                    {
                        "馬番": r.get("馬番"),
                        "EV": r.get("EV"),
                        "単勝オッズ": r.get("単勝オッズ"),
                    },
                    "単勝オッズ",
                )
            for r in tables.get("fukusho_ev", []):
                add_range("複勝", r, "複勝下限", "複勝上限", "EV下限", "EV上限")
            for r in tables.get("wide_ev", []):
                add_range("ワイド", r, "下限", "上限", "EV下限", "EV上限")

            if derived_rows:
                st.subheader("派生候補（参考: 単勝/複勝/ワイド：EV上位6三連複に含まれる馬番のみ）")
                render_copy_button(
                    "派生候補をコピー(TSV)", rows_to_tsv(derived_rows), key="derived_candidates"
                )
                st.dataframe(derived_rows, use_container_width=True)

            # 派生候補から「買い目に追加したもの（EV>=0）」を表示
            if str(pick_mode).startswith("確定枠") and derived_added_rows:
                added_table = [
                    {
                        "券種": r["種別"],
                        "買い目": r["買い目"],
                        "EV(またはEV上限)": float(r.get("EVU", 0.0)),
                        "当選確率(%)": round(float(r.get("P", 0.0)) * 100, 3),
                    }
                    for r in derived_added_rows
                ]
                st.subheader("追加買い目（派生候補から追加）")
                render_copy_button(
                    "追加買い目をコピー(TSV)", rows_to_tsv(added_table), key="derived_added"
                )
                st.dataframe(added_table, use_container_width=True)

        # --- 抽出結果（通常 or 三連複モード） ---
        if str(pick_mode).startswith("確定枠"):
            st.subheader("買い目（確定枠：ワイド(人気1-3) + 三連複EV上位5）")
        else:
            st.subheader("条件を満たす馬券（(EV≥x かつ 当選確率≥z) または 当選確率≥y）")
            st.caption(
                f"x={float(ev_and_threshold):g}, z={float(prob_threshold_percent):g}%, y={float(prob_or_threshold_percent):g}%"
            )
        if recs:
            detail_rows: List[Dict[str, Any]] = []
            usable_budget = (int(budget_yen) // 100) * 100
            for x in rec_rows:
                calc_fraction = float(
                    x.get("_BET_FRACTION", compute_bet_fraction(float(x["_KS"])))
                )
                stake_yen = int(x["掛け金(円)"])
                actual_fraction = 0.0 if usable_budget <= 0 else float(stake_yen) / float(usable_budget)
                pass_and = bool(x.get("_PASS_AND", False))
                pass_p_only = bool(x.get("_PASS_P_ONLY", False))
                detail_rows.append(
                    {
                        "種別": x["馬券"],
                        "買い目": x["買い目"],
                        "EV上限": round(float(x["_EVU"]), 4),
                        "当選確率(%)": round(float(x.get("_P", 0.0)) * 100, 3),
                        "条件(AND)": "○" if pass_and else "",
                        "条件(P≥y)": "○" if pass_p_only else "",
                        "KS(%)": round(float(x["_KS"]) * 100, 2),
                        "LS": round(float(x["_LS"]), 6),
                        "賭け割合(計算)(%)": round(calc_fraction * 100, 2),
                        "賭け割合(実)(%)": round(actual_fraction * 100, 2),
                        "賭け金(円)": stake_yen,
                    }
                )
            render_copy_button("抽出結果をコピー(TSV)", rows_to_tsv(detail_rows), key="filtered_table")
            st.dataframe(detail_rows, use_container_width=True)
        else:
            st.info("条件を満たす馬券はありませんでした。")

        st.subheader("EV：単勝")
        render_copy_button("EV：単勝をコピー(TSV)", rows_to_tsv(tables["tansho_ev"]), key="tansho_ev")
        st.dataframe(tables["tansho_ev"], use_container_width=True)

        st.subheader("EV：複勝（3着以内）")
        render_copy_button("EV：複勝をコピー(TSV)", rows_to_tsv(tables["fukusho_ev"]), key="fukusho_ev")
        st.dataframe(tables["fukusho_ev"], use_container_width=True)

        st.subheader("EV：馬単")
        render_copy_button("EV：馬単をコピー(TSV)", rows_to_tsv(tables["umatan_ev"]), key="umatan_ev")
        st.dataframe(tables["umatan_ev"], use_container_width=True)

        st.subheader("EV：三連単")
        render_copy_button("EV：三連単をコピー(TSV)", rows_to_tsv(tables["sanrentan_ev"]), key="sanrentan_ev")
        st.dataframe(tables["sanrentan_ev"], use_container_width=True)

        st.subheader("EV：馬連")
        render_copy_button("EV：馬連をコピー(TSV)", rows_to_tsv(tables["umaren_ev"]), key="umaren_ev")
        st.dataframe(tables["umaren_ev"], use_container_width=True)

        st.subheader("EV：三連複")
        render_copy_button("EV：三連複をコピー(TSV)", rows_to_tsv(tables["sanrenpuku_ev"]), key="sanrenpuku_ev")
        st.dataframe(tables["sanrenpuku_ev"], use_container_width=True)

        st.subheader("EV：ワイド")
        render_copy_button("EV：ワイドをコピー(TSV)", rows_to_tsv(tables["wide_ev"]), key="wide_ev")
        st.dataframe(tables["wide_ev"], use_container_width=True)

        # UI表示ルールにより odds.json の内容は表示しない（取得・保存は内部で維持）

    except Exception as e:
        st.session_state["_last_error"] = str(e)
        st.error(str(e))
    finally:
        st.session_state["_run_output"] = False

if st.session_state.get("_last_error"):
    # 直近のエラーを残しておく（リセットで消える）
    st.caption(f"直近エラー: {st.session_state.get('_last_error')}")
