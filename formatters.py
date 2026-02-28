#----------表示--------------
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from models import MarketProbRow


def format_market_table(rows: Iterable[MarketProbRow]) -> str:
    # 市場オッズから算出した基本テーブル（単勝・市場勝率）を文字列で整形
    header = ["馬番", "馬名", "単勝", "q=1/odds", "勝率(市場)"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(int(row.number)),
                    str(row.name),
                    f"{float(row.win_odds):.1f}",
                    f"{float(row.q):.6f}",
                    f"{float(row.market_win_prob) * 100:.2f}%",
                ]
            )
        )
    return "\n".join(lines) + "\n"


def format_pl_table(title: str, rows: Iterable[Mapping[str, Any]], order_keys: Sequence[str]) -> str:
    # 着順（PLモデル）確率テーブルを整形（馬単・三連単など）
    header = [title, "確率(PL)", "確率(PL)%", "オッズ"]
    lines = ["\t".join(header)]
    for row in rows:
        order = "→".join(str(int(row[k])) for k in order_keys)
        prob = float(row["pl_prob"])
        odds = row.get("odds")
        odds_str = "" if odds is None else f"{float(odds):.1f}"
        lines.append(f"{order}\t{prob:.6f}\t{prob * 100:.2f}%\t{odds_str}")
    return "\n".join(lines) + "\n"


def format_pair_table(title: str, rows: Iterable[Mapping[str, Any]]) -> str:
    # 馬連・ワイドなど「2頭組み合わせ」用のPL確率テーブル
    header = [title, "確率(PL)", "確率(PL)%", "オッズ", "下限", "上限"]
    lines = ["\t".join(header)]
    for row in rows:
        i = int(row["i"])
        j = int(row["j"])
        prob = float(row["pl_prob"])
        odds = row.get("odds")
        odds_str = "" if odds is None else f"{float(odds):.1f}"
        lo = row.get("odds_min")
        hi = row.get("odds_max")
        lo_str = "" if lo is None else f"{float(lo):.1f}"
        hi_str = "" if hi is None else f"{float(hi):.1f}"
        lines.append(f"{i}-{j}\t{prob:.6f}\t{prob * 100:.2f}%\t{odds_str}\t{lo_str}\t{hi_str}")
    return "\n".join(lines) + "\n"


def format_triple_table(title: str, rows: Iterable[Mapping[str, Any]]) -> str:
    # 三連系（3頭組）PL確率テーブル
    header = [title, "確率(PL)", "確率(PL)%", "オッズ"]
    lines = ["\t".join(header)]
    for row in rows:
        a = int(row["a"])
        b = int(row["b"])
        c = int(row["c"])
        prob = float(row["pl_prob"])
        odds = row.get("odds")
        odds_str = "" if odds is None else f"{float(odds):.1f}"
        lines.append(f"{a}-{b}-{c}\t{prob:.6f}\t{prob * 100:.2f}%\t{odds_str}")
    return "\n".join(lines) + "\n"


def format_fukusho_table(title: str, rows: Iterable[Mapping[str, Any]]) -> str:
    # 複勝（3着以内）確率テーブル
    header = [title, "馬番", "馬名", "確率(PL)", "確率(PL)%"]
    lines = ["\t".join(header)]
    for row in rows:
        number = int(row["number"])
        name = str(row["name"])
        prob = float(row["pl_prob"])
        lines.append(f"\t{number}\t{name}\t{prob:.6f}\t{prob * 100:.2f}%")
    return "\n".join(lines) + "\n"


def format_tansho_ev_table(rows: Iterable[Mapping[str, Any]]) -> str:
    # 単勝の期待値（EV）テーブル
    header = ["単勝", "馬番", "馬名", "当選確率", "単勝オッズ", "EV", "EV'", "p×log(o)", "Kelly(f*)"]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    "",
                    str(int(row["number"])),
                    str(row["name"]),
                    f'{float(row["prob"]) * 100:.2f}%',
                    f'{float(row["odds"]):.1f}',
                    f'{float(row["ev"]):.4f}',
                    f'{float(row["ev_prime"]):.6f}',
                    f'{float(row["p_log_o"]):.6f}',
                    f'{float(row["kelly"]) * 100:.2f}%',
                ]
            )
        )
    return "\n".join(lines) + "\n"


def format_fukusho_ev_table(rows: Iterable[Mapping[str, Any]]) -> str:
    # 複勝の期待値レンジ（EV下限〜上限）テーブル
    header = [
        "複勝(3着以内)",
        "馬番",
        "馬名",
        "当選確率",
        "複勝下限",
        "複勝上限",
        "EV下限",
        "EV上限",
        "EV'下限",
        "EV'上限",
        "p×log(o)下限",
        "p×log(o)上限",
        "Kelly下限",
        "Kelly上限",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    "",
                    str(int(row["number"])),
                    str(row["name"]),
                    f'{float(row["prob"]) * 100:.2f}%',
                    f'{float(row["odds_min"]):.1f}',
                    f'{float(row["odds_max"]):.1f}',
                    f'{float(row["ev_min"]):.4f}',
                    f'{float(row["ev_max"]):.4f}',
                    f'{float(row["ev_prime_min"]):.6f}',
                    f'{float(row["ev_prime_max"]):.6f}',
                    f'{float(row["p_log_o_min"]):.6f}',
                    f'{float(row["p_log_o_max"]):.6f}',
                    f'{float(row["kelly_min"]) * 100:.2f}%',
                    f'{float(row["kelly_max"]) * 100:.2f}%',
                ]
            )
        )
    return "\n".join(lines) + "\n"


def format_order_ev_table(title: str, rows: Iterable[Mapping[str, Any]], order_keys: Sequence[str]) -> str:
    # 着順指定（馬単・三連単）のEVテーブル
    header = [title, "当選確率", "当選確率%", "オッズ", "EV", "EV'", "p×log(o)", "Kelly(f*)"]
    lines = ["\t".join(header)]
    for row in rows:
        order = "→".join(str(int(row[k])) for k in order_keys)
        prob = float(row["prob"])
        odds = float(row["odds"])
        ev = float(row["ev"])
        ev_prime = float(row["ev_prime"])
        p_log_o = float(row["p_log_o"])
        kelly = float(row["kelly"])
        lines.append(
            f"{order}\t{prob:.6f}\t{prob * 100:.2f}%\t{odds:.1f}\t{ev:.4f}\t{ev_prime:.6f}\t"
            f"{p_log_o:.6f}\t{kelly * 100:.2f}%"
        )
    return "\n".join(lines) + "\n"


def format_pair_ev_table(title: str, rows: Iterable[Mapping[str, Any]]) -> str:
    # 馬連など2頭組み合わせのEVテーブル
    header = [title, "当選確率", "当選確率%", "オッズ", "EV", "EV'", "p×log(o)", "Kelly(f*)"]
    lines = ["\t".join(header)]
    for row in rows:
        i = int(row["i"])
        j = int(row["j"])
        prob = float(row["prob"])
        odds = float(row["odds"])
        ev = float(row["ev"])
        ev_prime = float(row["ev_prime"])
        p_log_o = float(row["p_log_o"])
        kelly = float(row["kelly"])
        lines.append(
            f"{i}-{j}\t{prob:.6f}\t{prob * 100:.2f}%\t{odds:.1f}\t{ev:.4f}\t{ev_prime:.6f}\t"
            f"{p_log_o:.6f}\t{kelly * 100:.2f}%"
        )
    return "\n".join(lines) + "\n"


def format_triple_ev_table(title: str, rows: Iterable[Mapping[str, Any]]) -> str:
    # 三連系（3頭）のEVテーブル
    header = [title, "当選確率", "当選確率%", "オッズ", "EV", "EV'", "p×log(o)", "Kelly(f*)"]
    lines = ["\t".join(header)]
    for row in rows:
        a = int(row["a"])
        b = int(row["b"])
        c = int(row["c"])
        prob = float(row["prob"])
        odds = float(row["odds"])
        ev = float(row["ev"])
        ev_prime = float(row["ev_prime"])
        p_log_o = float(row["p_log_o"])
        kelly = float(row["kelly"])
        lines.append(
            f"{a}-{b}-{c}\t{prob:.6f}\t{prob * 100:.2f}%\t{odds:.1f}\t{ev:.4f}\t{ev_prime:.6f}\t"
            f"{p_log_o:.6f}\t{kelly * 100:.2f}%"
        )
    return "\n".join(lines) + "\n"


def format_wide_ev_table(title: str, rows: Iterable[Mapping[str, Any]]) -> str:
    # ワイド用：オッズ幅を考慮したEVレンジテーブル
    header = [
        title,
        "当選確率",
        "当選確率%",
        "下限",
        "上限",
        "EV下限",
        "EV上限",
        "EV'下限",
        "EV'上限",
        "p×log(o)下限",
        "p×log(o)上限",
        "Kelly下限",
        "Kelly上限",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        i = int(row["i"])
        j = int(row["j"])
        prob = float(row["prob"])
        odds_min = float(row["odds_min"])
        odds_max = float(row["odds_max"])
        lines.append(
            f"{i}-{j}\t{prob:.6f}\t{prob * 100:.2f}%\t{odds_min:.1f}\t{odds_max:.1f}\t"
            f'{float(row["ev_min"]):.4f}\t{float(row["ev_max"]):.4f}\t'
            f'{float(row["ev_prime_min"]):.6f}\t{float(row["ev_prime_max"]):.6f}\t'
            f'{float(row["p_log_o_min"]):.6f}\t{float(row["p_log_o_max"]):.6f}\t'
            f'{float(row["kelly_min"]) * 100:.2f}%\t{float(row["kelly_max"]) * 100:.2f}%'
        )
    return "\n".join(lines) + "\n"
