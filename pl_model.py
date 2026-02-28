#----------着順の確率計算-----------
from __future__ import annotations

from typing import Mapping, Sequence


def plackett_luce_order_probability(
    prob_by_number: Mapping[int, float],
    order: Sequence[int],
) -> float:
    # Plackett–Luce（PL）モデルで、指定された着順そのものの確率を計算する
    # 各着順で「まだ残っている馬の中から選ばれる確率」を掛け合わせる
    used: set[int] = set()
    remaining_total = 1.0  # 未選択馬の確率合計
    probability = 1.0

    for position, number in enumerate(order):
        # 同じ馬が複数回出てきたらエラー
        if number in used:
            raise ValueError(f"Order has duplicate number at position {position}: {number}")

        # 馬番に対応する確率を取得
        p = prob_by_number.get(number)
        if p is None:
            raise ValueError(f"Unknown horse number in order: {number}")
        if p < 0:
            raise ValueError(f"Invalid probability (negative) for number={number}")

        # 残り確率が尽きたら以降の確率は0
        if remaining_total <= 0:
            return 0.0

        # 「残っている中でその馬が来る確率」を掛ける
        probability *= p / remaining_total

        # 使用済みにして残り確率から差し引く
        used.add(number)
        remaining_total -= p

    return probability


def parse_number_list(value: str) -> list[int]:
    # "1,3,5" のような文字列を [1, 3, 5] に変換する
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty order.")
    return [int(p) for p in parts]
