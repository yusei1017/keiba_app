#----------EV計算--------------
from __future__ import annotations

import math
from typing import Mapping


def compute_ev(prob: float, odds: float) -> float:
    # 期待値（EV）を計算する
    # EV = 的中確率 × オッズ − 1（購入金額を1とした場合）
    return prob * odds - 1.0


def compute_ev_range(prob: float, odds_min: float, odds_max: float) -> Mapping[str, float]:
    # オッズの最小〜最大を想定した期待値レンジを計算する
    return {
        "ev_min": compute_ev(prob, odds_min),  # 最低オッズ時のEV
        "ev_max": compute_ev(prob, odds_max),  # 最高オッズ時のEV
    }


def compute_ev_prime(prob: float, odds: float) -> float:
    # EV' を計算する
    # EV' = EV × 的中確率
    return compute_ev(prob, odds) * prob


def compute_ev_prime_range(prob: float, odds_min: float, odds_max: float) -> Mapping[str, float]:
    # オッズの最小〜最大を想定した EV' レンジを計算する
    evs = compute_ev_range(prob, odds_min, odds_max)
    return {
        "ev_prime_min": float(evs["ev_min"]) * prob,
        "ev_prime_max": float(evs["ev_max"]) * prob,
    }


def compute_p_log_o(prob: float, odds: float) -> float:
    # p × log(o)（対数は自然対数）
    if odds <= 0:
        return float("nan")
    return prob * math.log(odds)


def compute_kelly_fraction(prob: float, odds: float) -> float:
    """
    ケリー基準の最適ベット比率（小数オッズ）:
      f* = (p*odds - 1) / (odds - 1)
    """
    if odds <= 1:
        return 0.0
    return (prob * odds - 1.0) / (odds - 1.0)


def clip_fraction(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if value < low:
        return low
    if value > high:
        return high
    return value


def compute_kelly_fraction_clipped(prob: float, odds: float) -> float:
    # 現実運用しやすいよう 0〜1 に丸める（マイナスは買わない）
    return clip_fraction(compute_kelly_fraction(prob, odds), 0.0, 1.0)


def compute_kelly_range(prob: float, odds_min: float, odds_max: float) -> Mapping[str, float]:
    return {
        "kelly_min": compute_kelly_fraction_clipped(prob, odds_min),
        "kelly_max": compute_kelly_fraction_clipped(prob, odds_max),
    }


def compute_p_log_o_range(prob: float, odds_min: float, odds_max: float) -> Mapping[str, float]:
    return {
        "p_log_o_min": compute_p_log_o(prob, odds_min),
        "p_log_o_max": compute_p_log_o(prob, odds_max),
    }
