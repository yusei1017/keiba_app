#----------勝率計算--------------
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from models import MarketProbRow, TanshoEntry


def compute_market_win_probabilities(
    entries: Sequence[TanshoEntry],
    weights_rank: Sequence[float] | None = None,
) -> list[MarketProbRow]:
    """
    市場確率（勝率）を「単勝オッズの逆数」を正規化して作ります。

    数式:
      q_i = 1 / odds_i
      p_i = q_i / Σ_j q_j

    ※互換のため weights_rank 引数は残していますが、この方式では使用しません。
    """
    if not entries:
        return []

    qs: list[float] = []
    for e in entries:
        o = float(e.win_odds)
        if o <= 0:
            raise ValueError("win_odds must be > 0.")
        qs.append(1.0 / o)
    total_q = sum(qs)
    if total_q <= 0:
        raise ValueError("Sum of inverse odds must be > 0.")

    rows: list[MarketProbRow] = []
    for entry, q in zip(entries, qs):
        p = float(q) / float(total_q)
        rows.append(
            MarketProbRow(
                number=entry.number,
                name=entry.name,
                win_odds=entry.win_odds,
                q=float(q),  # q = 1/odds
                market_win_prob=p,
            )
        )
    return rows


def build_prob_by_number(rows: Iterable[MarketProbRow]) -> Mapping[int, float]:
    # 馬番 → 市場勝率 の辞書を作る
    return {int(row.number): float(row.market_win_prob) for row in rows}


def build_name_by_number(rows: Iterable[MarketProbRow]) -> Mapping[int, str]:
    # 馬番 → 馬名 の辞書を作る
    return {int(row.number): str(row.name) for row in rows}
