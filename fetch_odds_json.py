#----------URLからHTML取得してjsonに変換する--------------
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import requests
from bs4 import BeautifulSoup


# ----------------------------
# Step A: URL -> HTML
# ----------------------------
def fetch_html(url: str, timeout: int = 15) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=timeout)

    if r.status_code in (403, 429):
        raise RuntimeError(
            f"Fetch blocked (status={r.status_code}). "
            "Do not retry repeatedly; wait and try later."
        )
    r.raise_for_status()

    enc = r.encoding or r.apparent_encoding or "utf-8"
    return r.content.decode(enc, errors="replace")


# ----------------------------
# Utils
# ----------------------------
_RANGE_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)\s*$")


def parse_range(s: str) -> Dict[str, float]:
    """
    "1.0 - 1.2" -> {"low": 1.0, "high": 1.2}
    """
    m = _RANGE_RE.match(s.strip())
    if not m:
        raise ValueError(f"range parse failed: {s}")
    return {"low": float(m.group(1)), "high": float(m.group(2))}


def join_key(nums: List[str], sep: str) -> str:
    return sep.join(nums)


# ----------------------------
# Step C': Parse tables -> odds.json
# ----------------------------
def parse_horses_table(table) -> Dict[str, Any]:
    """
    単勝・複勝（馬一覧）テーブルから horses と win/place を作るための情報を抜く。
    """
    horses: Dict[str, Any] = {}
    rows = table.find_all("tr")
    for tr in rows[1:]:  # skip header
        tds = tr.find_all("td")
        if len(tds) < 7:
            continue

        # 3列目が馬番（このページの構造）
        horse_no = tds[2].get_text(strip=True)
        name = tds[4].get_text(strip=True)

        win_odds = tds[5].get_text(strip=True)
        place_odds = tds[6].get_text(strip=True)

        if not horse_no or not win_odds:
            continue

        horses[str(horse_no)] = {
            "name": name,
            "win_odds": float(win_odds),
            "place_odds": parse_range(place_odds),
        }
    return horses


def parse_combi_table(table) -> Tuple[str, Dict[str, Any]]:
    """
    2頭/3頭/三連単(矢印)の各テーブルを判別して tickets の一部を返す。
    返り値: (ticket_kind, mapping)
      - "trifecta": {"5>9>14": 27.4, ...}
      - "exacta": {"5>9": 6.2, ...}
      - "quinella": {"5-9": 4.2, ...}
      - "wide": {"5-9": {"low":2.0,"high":2.3}, ...}
      - "trio": {"5-9-14": 8.2, ...}
    """
    rows = table.find_all("tr")
    if len(rows) <= 1:
        return ("unknown", {})

    mapping: Dict[str, Any] = {}

    # 1行目（ヘッダ）で「ワイド列があるか」ざっくり判定
    header_text = rows[0].get_text(" ", strip=True)
    has_wide_col = "ワイド" in header_text or "¥ï¥¤¥É" in header_text  # 文字化け対策（wideっぽい）

    # 2行目以降の最初の行を見て、矢印かハイフンか、頭数を判別
    sample = rows[1]
    nums = [x.get_text(strip=True) for x in sample.select("span.UmaBan")]
    has_arrow = sample.select_one("span.Kaime_Arrow") is not None
    has_hyphen = sample.select_one("span.Hyphen") is not None

    # 種別推定
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

    # 行をなめて詰める
    for tr in rows[1:]:
        nums = [x.get_text(strip=True) for x in tr.select("span.UmaBan")]
        if not nums:
            continue

        odds_tds = tr.find_all("td", class_="Odds")
        # 三連単/馬単/馬連/三連複：odds_tds[-1] がオッズのことが多い
        # 馬連ワイド：odds_tds が2つ（馬連, ワイド）
        if kind == "quinella_wide":
            if len(nums) != 2 or len(odds_tds) < 2:
                continue
            key = join_key(nums, "-")
            quinella_odds = float(odds_tds[0].get_text(strip=True))
            wide_range = parse_range(odds_tds[1].get_text(strip=True))
            # 両方返すため、ここでは特殊扱い（後で分配する）
            mapping[key] = {"quinella": quinella_odds, "wide": wide_range}
        else:
            if kind in ("trifecta", "exacta"):
                sep = ">"
            else:
                sep = "-"
            key = join_key(nums, sep)
            if not odds_tds:
                continue
            odds = float(odds_tds[-1].get_text(strip=True))
            mapping[key] = odds

    return (kind, mapping)


def build_odds_json(url: str, html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    tables = soup.find_all("table", class_="RaceOdds_HorseList_Table")
    if not tables:
        raise RuntimeError("No tables found. Page may be dynamically generated or blocked.")

    horses: Dict[str, Any] = {}
    tickets: Dict[str, Any] = {
        "win": {},
        "place": {},
        "exacta": {},
        "quinella": {},
        "wide": {},
        "trifecta": {},
        "trio": {},
    }

    # まず馬一覧っぽいテーブル（Horse_Name列がある）を探す
    horses_table = None
    for t in tables:
        if t.select_one("th.Horse_Name") is not None:
            horses_table = t
            break
    if horses_table is None:
        raise RuntimeError("Horses table not found.")

    horses = parse_horses_table(horses_table)

    # horses から win/place を展開
    for no, info in horses.items():
        tickets["win"][no] = float(info["win_odds"])
        tickets["place"][no] = info["place_odds"]

    # それ以外のテーブルを券種として回収
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
            # mapping: {"5-9": {"quinella": 4.2, "wide": {"low":..,"high":..}}, ...}
            for key, v in mapping.items():
                tickets["quinella"][key] = v["quinella"]
                tickets["wide"][key] = v["wide"]

    out = {
        "meta": {
            "source_url": url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
        "horses": horses,
        "tickets": tickets,
    }
    return out


# ----------------------------
# CLI
# ----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", default="odds.json")
    args = ap.parse_args()

    try:
        html = fetch_html(args.url)
        odds = build_odds_json(args.url, html)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(odds, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved: {args.out}")
    print(f"[INFO] horses={len(odds['horses'])}")
    print(f"[INFO] exacta={len(odds['tickets']['exacta'])}, "
          f"quinella={len(odds['tickets']['quinella'])}, wide={len(odds['tickets']['wide'])}, "
          f"trio={len(odds['tickets']['trio'])}, trifecta={len(odds['tickets']['trifecta'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
