#----------JSONデータの読み込みと変換--------------
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from models import (
    OddsData,
    SanrenpukuOdds,
    SanrentanOdds,
    TanshoEntry,
    TanshoFukushoOdds,
    UmarenOdds,
    UmatanOdds,
    WideOdds,
)


def load_json(path: Path) -> Any:
    # JSONファイルを読み込んでPythonオブジェクトに変換
    return json.loads(path.read_text(encoding="utf-8"))


def _require_mapping(value: Any, message: str) -> Mapping[str, Any]:
    # dict形式であることを保証するためのチェック
    if not isinstance(value, Mapping):
        raise ValueError(message)
    return value


def _require_sequence(value: Any, message: str) -> Sequence[Any]:
    # list/tuple形式であることを保証（文字列は除外）
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(message)
    return value


def parse_odds_data(data: Any) -> OddsData:
    root = _require_mapping(data, "odds JSON must be an object.")

    # odds.json（新形式）と odds_sample.json（旧形式）の両方を受け付けます。
    # 新形式:
    #   {"horses": {...}, "tickets": {...}, "meta": {...}}
    # 旧形式:
    #   {"tansho_fukusho": [...], "umatan": [...], ...}

    # --- 新形式（odds.json） ---
    if "horses" in root and "tickets" in root:
        horses = _require_mapping(root["horses"], '"horses" must be an object.')
        tickets = _require_mapping(root["tickets"], '"tickets" must be an object.')

        win_map: Mapping[str, Any] = {}
        place_map: Mapping[str, Any] = {}
        if "win" in tickets:
            win_map = _require_mapping(tickets["win"], '"tickets.win" must be an object.')
        if "place" in tickets:
            place_map = _require_mapping(tickets["place"], '"tickets.place" must be an object.')

        tansho_fukusho: list[TanshoFukushoOdds] = []
        for number_str in sorted(horses.keys(), key=lambda s: int(s)):
            horse_obj = _require_mapping(
                horses[number_str], f'"horses"."{number_str}" must be an object.'
            )
            number = int(number_str)

            win_odds = horse_obj.get("win_odds", win_map.get(number_str))
            if win_odds is None:
                raise ValueError(f'Missing win_odds for horse "{number_str}".')

            place_obj = horse_obj.get("place_odds", place_map.get(number_str))
            place_obj = _require_mapping(place_obj, f'Missing place_odds for horse "{number_str}".')
            low = place_obj.get("low")
            high = place_obj.get("high")
            if low is None or high is None:
                raise ValueError(f'place_odds must have low/high for horse "{number_str}".')

            tansho_fukusho.append(
                TanshoFukushoOdds(
                    number=number,
                    name=str(horse_obj.get("name", "")),
                    win_odds=float(win_odds),
                    place_odds_min=float(low),
                    place_odds_max=float(high),
                )
            )

        umatan: list[UmatanOdds] = []
        if "exacta" in tickets:
            exacta = _require_mapping(tickets["exacta"], '"tickets.exacta" must be an object.')
            for key, odds in exacta.items():
                a_str, b_str = str(key).split(">", 1)
                umatan.append(UmatanOdds(first=int(a_str), second=int(b_str), odds=float(odds)))

        sanrentan: list[SanrentanOdds] = []
        if "trifecta" in tickets:
            trifecta = _require_mapping(tickets["trifecta"], '"tickets.trifecta" must be an object.')
            for key, odds in trifecta.items():
                a_str, b_str, c_str = str(key).split(">", 2)
                sanrentan.append(
                    SanrentanOdds(
                        first=int(a_str), second=int(b_str), third=int(c_str), odds=float(odds)
                    )
                )

        umaren: list[UmarenOdds] = []
        if "quinella" in tickets:
            quinella = _require_mapping(tickets["quinella"], '"tickets.quinella" must be an object.')
            for key, odds in quinella.items():
                a_str, b_str = str(key).split("-", 1)
                umaren.append(UmarenOdds(i=int(a_str), j=int(b_str), odds=float(odds)))

        wide: list[WideOdds] = []
        if "wide" in tickets:
            wide_map = _require_mapping(tickets["wide"], '"tickets.wide" must be an object.')
            for key, value in wide_map.items():
                a_str, b_str = str(key).split("-", 1)
                obj = _require_mapping(value, f'"tickets.wide"."{key}" must be an object.')
                wide.append(
                    WideOdds(
                        i=int(a_str),
                        j=int(b_str),
                        odds_min=float(obj["low"]),
                        odds_max=float(obj["high"]),
                    )
                )

        sanrenpuku: list[SanrenpukuOdds] = []
        if "trio" in tickets:
            trio = _require_mapping(tickets["trio"], '"tickets.trio" must be an object.')
            for key, odds in trio.items():
                a_str, b_str, c_str = str(key).split("-", 2)
                sanrenpuku.append(
                    SanrenpukuOdds(a=int(a_str), b=int(b_str), c=int(c_str), odds=float(odds))
                )

        return OddsData(
            tansho_fukusho=tansho_fukusho,
            umatan=umatan,
            sanrentan=sanrentan,
            umaren=umaren,
            wide=wide,
            sanrenpuku=sanrenpuku,
        )

    # --- 旧形式（odds_sample.json） ---
    tf_raw = _require_sequence(
        root.get("tansho_fukusho"), 'JSON must contain key "tansho_fukusho" as array.'
    )
    tansho_fukusho_legacy: list[TanshoFukushoOdds] = []
    for idx, item in enumerate(tf_raw):
        obj = _require_mapping(item, f'"tansho_fukusho"[{idx}] must be an object.')
        tansho_fukusho_legacy.append(
            TanshoFukushoOdds(
                number=int(obj["number"]),
                name=str(obj["name"]),
                win_odds=float(obj["win_odds"]),
                place_odds_min=float(obj["place_odds_min"]),
                place_odds_max=float(obj["place_odds_max"]),
            )
        )

    umatan_legacy: list[UmatanOdds] = []
    if "umatan" in root:
        for idx, item in enumerate(_require_sequence(root["umatan"], '"umatan" must be an array.')):
            obj = _require_mapping(item, f'"umatan"[{idx}] must be an object.')
            umatan_legacy.append(
                UmatanOdds(first=int(obj["first"]), second=int(obj["second"]), odds=float(obj["odds"]))
            )

    sanrentan_legacy: list[SanrentanOdds] = []
    if "sanrentan" in root:
        for idx, item in enumerate(
            _require_sequence(root["sanrentan"], '"sanrentan" must be an array.')
        ):
            obj = _require_mapping(item, f'"sanrentan"[{idx}] must be an object.')
            sanrentan_legacy.append(
                SanrentanOdds(
                    first=int(obj["first"]),
                    second=int(obj["second"]),
                    third=int(obj["third"]),
                    odds=float(obj["odds"]),
                )
            )

    umaren_legacy: list[UmarenOdds] = []
    if "umaren" in root:
        for idx, item in enumerate(_require_sequence(root["umaren"], '"umaren" must be an array.')):
            obj = _require_mapping(item, f'"umaren"[{idx}] must be an object.')
            pair = _require_sequence(obj.get("pair"), f'"umaren"[{idx}].pair must be [i,j].')
            if len(pair) != 2:
                raise ValueError(f'"umaren"[{idx}].pair must be length 2.')
            umaren_legacy.append(UmarenOdds(i=int(pair[0]), j=int(pair[1]), odds=float(obj["odds"])))

    wide_legacy: list[WideOdds] = []
    if "wide" in root:
        for idx, item in enumerate(_require_sequence(root["wide"], '"wide" must be an array.')):
            obj = _require_mapping(item, f'"wide"[{idx}] must be an object.')
            pair = _require_sequence(obj.get("pair"), f'"wide"[{idx}].pair must be [i,j].')
            if len(pair) != 2:
                raise ValueError(f'"wide"[{idx}].pair must be length 2.')
            wide_legacy.append(
                WideOdds(
                    i=int(pair[0]),
                    j=int(pair[1]),
                    odds_min=float(obj["odds_min"]),
                    odds_max=float(obj["odds_max"]),
                )
            )

    sanrenpuku_legacy: list[SanrenpukuOdds] = []
    if "sanrenpuku" in root:
        for idx, item in enumerate(
            _require_sequence(root["sanrenpuku"], '"sanrenpuku" must be an array.')
        ):
            obj = _require_mapping(item, f'"sanrenpuku"[{idx}] must be an object.')
            combo = _require_sequence(obj.get("combo"), f'"sanrenpuku"[{idx}].combo must be [a,b,c].')
            if len(combo) != 3:
                raise ValueError(f'"sanrenpuku"[{idx}].combo must be length 3.')
            sanrenpuku_legacy.append(
                SanrenpukuOdds(
                    a=int(combo[0]),
                    b=int(combo[1]),
                    c=int(combo[2]),
                    odds=float(obj["odds"]),
                )
            )

    return OddsData(
        tansho_fukusho=tansho_fukusho_legacy,
        umatan=umatan_legacy,
        sanrentan=sanrentan_legacy,
        umaren=umaren_legacy,
        wide=wide_legacy,
        sanrenpuku=sanrenpuku_legacy,
    )


def parse_tansho_entries(odds: OddsData) -> list[TanshoEntry]:
    # 単勝計算用に必要な最低限の情報だけを抽出
    return [
        TanshoEntry(number=item.number, name=item.name, win_odds=item.win_odds)
        for item in odds.tansho_fukusho
    ]
