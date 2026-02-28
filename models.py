#----------型の統一--------------
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TanshoEntry:
    # 単勝計算に必要な最小単位の馬データ
    number: int
    name: str
    win_odds: float


@dataclass(frozen=True)
class TanshoFukushoOdds:
    # 単勝・複勝オッズをまとめた元データ
    number: int
    name: str
    win_odds: float
    place_odds_min: float
    place_odds_max: float


@dataclass(frozen=True)
class UmatanOdds:
    # 馬単（1着→2着）のオッズ
    first: int
    second: int
    odds: float


@dataclass(frozen=True)
class SanrentanOdds:
    # 三連単（1着→2着→3着）のオッズ
    first: int
    second: int
    third: int
    odds: float


@dataclass(frozen=True)
class UmarenOdds:
    # 馬連（順不同2頭）のオッズ
    i: int
    j: int
    odds: float


@dataclass(frozen=True)
class WideOdds:
    # ワイド（順不同2頭・オッズ幅あり）
    i: int
    j: int
    odds_min: float
    odds_max: float


@dataclass(frozen=True)
class SanrenpukuOdds:
    # 三連複（順不同3頭）のオッズ
    a: int
    b: int
    c: int
    odds: float


@dataclass(frozen=True)
class OddsData:
    # 全券種のオッズをまとめたコンテナ
    tansho_fukusho: List[TanshoFukushoOdds]
    umatan: List[UmatanOdds]
    sanrentan: List[SanrentanOdds]
    umaren: List[UmarenOdds]
    wide: List[WideOdds]
    sanrenpuku: List[SanrenpukuOdds]


@dataclass(frozen=True)
class MarketProbRow:
    # 市場オッズから算出した勝率を保持する行データ
    number: int
    name: str
    win_odds: float
    q: float                 # q = 1 / win_odds
    market_win_prob: float   # 市場が織り込む勝率
