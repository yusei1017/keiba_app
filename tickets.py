#----------馬券ごとの当選確率の計算--------------
from __future__ import annotations

import itertools
from typing import Mapping, Sequence

from pl_model import plackett_luce_order_probability


def umatan_probability(prob_by_number: Mapping[int, float], first: int, second: int) -> float:
    # 馬単（1着→2着）の当選確率
    return plackett_luce_order_probability(prob_by_number, [first, second])


def sanrentan_probability(
    prob_by_number: Mapping[int, float],
    first: int,
    second: int,
    third: int,
) -> float:
    # 三連単（1着→2着→3着）の当選確率
    return plackett_luce_order_probability(prob_by_number, [first, second, third])


def umaren_probability(prob_by_number: Mapping[int, float], i: int, j: int) -> float:
    # 馬連（順不同2頭）なので i→j と j→i を合算
    return (
        plackett_luce_order_probability(prob_by_number, [i, j])
        + plackett_luce_order_probability(prob_by_number, [j, i])
    )


def sanrenpuku_probability(prob_by_number: Mapping[int, float], a: int, b: int, c: int) -> float:
    # 三連複（順不同3頭）：全順列の確率を合算
    total = 0.0
    for perm in itertools.permutations([a, b, c], 3):
        total += plackett_luce_order_probability(prob_by_number, perm)
    return total


def wide_probability(
    prob_by_number: Mapping[int, float],
    all_numbers: Sequence[int],
    i: int,
    j: int,
) -> float:
    # ワイド：i・jが3着以内に入る確率
    # = i・j・k の三連複確率を全kについて合算
    if len(all_numbers) < 3:
        raise ValueError("Wide requires at least 3 horses.")

    total = 0.0
    for k in all_numbers:
        if k == i or k == j:
            continue
        total += sanrenpuku_probability(prob_by_number, i, j, k)
    return total


def fukusho_probability(
    prob_by_number: Mapping[int, float],
    all_numbers: Sequence[int],
    number: int,
) -> float:
    # 複勝（3着以内）に入る確率
    if len(all_numbers) < 3:
        raise ValueError("Fukusho requires at least 3 horses.")

    p_i = prob_by_number.get(number)
    if p_i is None:
        raise ValueError(f"Unknown horse number: {number}")

    # 1着になる確率
    p_first = float(p_i)

    # 2着になる確率（誰かが1着、その後にnumber）
    p_second = 0.0
    for j in all_numbers:
        if j == number:
            continue
        p_second += plackett_luce_order_probability(prob_by_number, [j, number])

    # 3着になる確率（誰かが1着・2着、その後にnumber）
    p_third = 0.0
    for j in all_numbers:
        if j == number:
            continue
        for k in all_numbers:
            if k == number or k == j:
                continue
            p_third += plackett_luce_order_probability(prob_by_number, [j, k, number])

    return p_first + p_second + p_third
