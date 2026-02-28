import argparse
import json
from pathlib import Path

import formatters
import io_json
import market_prob
import pl_model
import tickets
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
from models import OddsData


def _print_ev_all(
    odds: OddsData,
    prob_by_number: dict[int, float],
    all_numbers: list[int],
    name_by_number: dict[int, str],
) -> None:
    # 単勝（当選確率=market_win_prob）
    tansho_rows: list[dict] = []
    for item in odds.tansho_fukusho:
        p = float(prob_by_number[item.number])
        tansho_rows.append(
            {
                "number": item.number,
                "name": name_by_number.get(item.number, item.name),
                "prob": p,
                "odds": float(item.win_odds),
                "ev": compute_ev(p, float(item.win_odds)),
                "ev_prime": compute_ev_prime(p, float(item.win_odds)),
                "p_log_o": compute_p_log_o(p, float(item.win_odds)),
                "kelly": compute_kelly_fraction_clipped(p, float(item.win_odds)),
            }
        )
    tansho_rows.sort(key=lambda r: float(r["ev"]), reverse=True)
    print(formatters.format_tansho_ev_table(tansho_rows), end="\n")

    # 複勝（3着以内）＋複勝オッズレンジ
    fukusho_rows: list[dict] = []
    for item in odds.tansho_fukusho:
        p = tickets.fukusho_probability(prob_by_number, all_numbers, item.number)
        evs = compute_ev_range(p, float(item.place_odds_min), float(item.place_odds_max))
        evs_prime = compute_ev_prime_range(p, float(item.place_odds_min), float(item.place_odds_max))
        kellys = compute_kelly_range(p, float(item.place_odds_min), float(item.place_odds_max))
        p_logs = compute_p_log_o_range(p, float(item.place_odds_min), float(item.place_odds_max))
        fukusho_rows.append(
            {
                "number": item.number,
                "name": name_by_number.get(item.number, item.name),
                "prob": p,
                "odds_min": float(item.place_odds_min),
                "odds_max": float(item.place_odds_max),
                **evs,
                **evs_prime,
                **kellys,
                **p_logs,
            }
        )
    fukusho_rows.sort(key=lambda r: float(r["ev_max"]), reverse=True)
    print(formatters.format_fukusho_ev_table(fukusho_rows), end="\n")

    # 馬単
    if odds.umatan:
        rows = []
        for item in odds.umatan:
            p = tickets.umatan_probability(prob_by_number, item.first, item.second)
            rows.append(
                {
                    "first": item.first,
                    "second": item.second,
                    "prob": p,
                    "odds": float(item.odds),
                    "ev": compute_ev(p, float(item.odds)),
                    "ev_prime": compute_ev_prime(p, float(item.odds)),
                    "p_log_o": compute_p_log_o(p, float(item.odds)),
                    "kelly": compute_kelly_fraction_clipped(p, float(item.odds)),
                }
            )
        rows.sort(key=lambda r: float(r["ev"]), reverse=True)
        print(formatters.format_order_ev_table("馬単(1→2)", rows, ["first", "second"]), end="\n")

    # 三連単
    if odds.sanrentan:
        rows = []
        for item in odds.sanrentan:
            p = tickets.sanrentan_probability(prob_by_number, item.first, item.second, item.third)
            rows.append(
                {
                    "first": item.first,
                    "second": item.second,
                    "third": item.third,
                    "prob": p,
                    "odds": float(item.odds),
                    "ev": compute_ev(p, float(item.odds)),
                    "ev_prime": compute_ev_prime(p, float(item.odds)),
                    "p_log_o": compute_p_log_o(p, float(item.odds)),
                    "kelly": compute_kelly_fraction_clipped(p, float(item.odds)),
                }
            )
        rows.sort(key=lambda r: float(r["ev"]), reverse=True)
        print(
            formatters.format_order_ev_table("三連単(1→2→3)", rows, ["first", "second", "third"]),
            end="\n",
        )

    # 馬連
    if odds.umaren:
        rows = []
        for item in odds.umaren:
            p = tickets.umaren_probability(prob_by_number, item.i, item.j)
            rows.append(
                {
                    "i": item.i,
                    "j": item.j,
                    "prob": p,
                    "odds": float(item.odds),
                    "ev": compute_ev(p, float(item.odds)),
                    "ev_prime": compute_ev_prime(p, float(item.odds)),
                    "p_log_o": compute_p_log_o(p, float(item.odds)),
                    "kelly": compute_kelly_fraction_clipped(p, float(item.odds)),
                }
            )
        rows.sort(key=lambda r: float(r["ev"]), reverse=True)
        print(formatters.format_pair_ev_table("馬連(i-j)", rows), end="\n")

    # 三連複
    if odds.sanrenpuku:
        rows = []
        for item in odds.sanrenpuku:
            p = tickets.sanrenpuku_probability(prob_by_number, item.a, item.b, item.c)
            rows.append(
                {
                    "a": item.a,
                    "b": item.b,
                    "c": item.c,
                    "prob": p,
                    "odds": float(item.odds),
                    "ev": compute_ev(p, float(item.odds)),
                    "ev_prime": compute_ev_prime(p, float(item.odds)),
                    "p_log_o": compute_p_log_o(p, float(item.odds)),
                    "kelly": compute_kelly_fraction_clipped(p, float(item.odds)),
                }
            )
        rows.sort(key=lambda r: float(r["ev"]), reverse=True)
        print(formatters.format_triple_ev_table("三連複(a-b-c)", rows), end="\n")

    # ワイド（オッズレンジ）
    if odds.wide:
        rows = []
        for item in odds.wide:
            p = tickets.wide_probability(prob_by_number, all_numbers, item.i, item.j)
            evs = compute_ev_range(p, float(item.odds_min), float(item.odds_max))
            evs_prime = compute_ev_prime_range(p, float(item.odds_min), float(item.odds_max))
            kellys = compute_kelly_range(p, float(item.odds_min), float(item.odds_max))
            p_logs = compute_p_log_o_range(p, float(item.odds_min), float(item.odds_max))
            rows.append(
                {
                    "i": item.i,
                    "j": item.j,
                    "prob": p,
                    "odds_min": float(item.odds_min),
                    "odds_max": float(item.odds_max),
                    **evs,
                    **evs_prime,
                    **kellys,
                    **p_logs,
                }
            )
        rows.sort(key=lambda r: float(r["ev_max"]), reverse=True)
        print(formatters.format_wide_ev_table("ワイド(i-j)", rows), end="")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute market-implied probabilities, Plackett–Luce probabilities, and EV."
    )
    parser.add_argument(
        "odds_json",
        nargs="?",
        default="odds.json",
        help='Input JSON path (default: "odds.json").',
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format for market win probabilities.",
    )
    parser.add_argument(
        "--write",
        metavar="PATH",
        help='Write market probabilities JSON to PATH (only when --format=json).',
    )
    parser.add_argument(
        "--weights-rank",
        default="2.5,1.7,1.25,1.0,0.3,0.2",
        help="人気1..Nの重み（カンマ区切り）。例: 2.5,1.7,1.25,1.0,0.3,0.2",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pl-seq", metavar="ORDER", help='PLで順番確率（例: "5,9,14"）')
    mode.add_argument("--pl-umatan", action="store_true", help="umatan を PL確率表示")
    mode.add_argument("--pl-sanrentan", action="store_true", help="sanrentan を PL確率表示")
    mode.add_argument("--pl-umaren", action="store_true", help="umaren を PL確率表示")
    mode.add_argument("--pl-sanrenpuku", action="store_true", help="sanrenpuku を PL確率表示")
    mode.add_argument("--pl-wide", action="store_true", help="wide を PL確率表示")
    mode.add_argument("--pl-fukusho", action="store_true", help="複勝(3着以内) を PL確率表示")
    mode.add_argument("--ev-all", action="store_true", help="全券種の当選確率とEVを表示")

    args = parser.parse_args()

    data = io_json.load_json(Path(args.odds_json))
    odds = io_json.parse_odds_data(data)

    entries = io_json.parse_tansho_entries(odds)
    weights_rank = [float(x.strip()) for x in str(args.weights_rank).split(",") if x.strip()]
    market_rows = market_prob.compute_market_win_probabilities(entries, weights_rank=weights_rank)
    prob_by_number = dict(market_prob.build_prob_by_number(market_rows))
    name_by_number = dict(market_prob.build_name_by_number(market_rows))
    all_numbers = list(prob_by_number.keys())

    if args.pl_seq:
        order = pl_model.parse_number_list(args.pl_seq)
        p = pl_model.plackett_luce_order_probability(prob_by_number, order)
        print(f"ORDER\t{args.pl_seq}")
        print(f"PL_PROB\t{p:.10f}")
        print(f"PL_PROB_%\t{p * 100:.4f}%")
        return 0

    if args.pl_umatan:
        rows = [
            {
                "first": item.first,
                "second": item.second,
                "odds": item.odds,
                "pl_prob": pl_model.plackett_luce_order_probability(
                    prob_by_number, [item.first, item.second]
                ),
            }
            for item in odds.umatan
        ]
        print(formatters.format_pl_table("馬単(1→2)", rows, ["first", "second"]), end="")
        return 0

    if args.pl_sanrentan:
        rows = [
            {
                "first": item.first,
                "second": item.second,
                "third": item.third,
                "odds": item.odds,
                "pl_prob": pl_model.plackett_luce_order_probability(
                    prob_by_number, [item.first, item.second, item.third]
                ),
            }
            for item in odds.sanrentan
        ]
        print(formatters.format_pl_table("三連単(1→2→3)", rows, ["first", "second", "third"]), end="")
        return 0

    if args.pl_umaren:
        rows = [
            {
                "i": item.i,
                "j": item.j,
                "odds": item.odds,
                "pl_prob": tickets.umaren_probability(prob_by_number, item.i, item.j),
            }
            for item in odds.umaren
        ]
        print(formatters.format_pair_table("馬連(i-j)", rows), end="")
        return 0

    if args.pl_sanrenpuku:
        rows = [
            {
                "a": item.a,
                "b": item.b,
                "c": item.c,
                "odds": item.odds,
                "pl_prob": tickets.sanrenpuku_probability(prob_by_number, item.a, item.b, item.c),
            }
            for item in odds.sanrenpuku
        ]
        print(formatters.format_triple_table("三連複(a-b-c)", rows), end="")
        return 0

    if args.pl_wide:
        rows = [
            {
                "i": item.i,
                "j": item.j,
                "odds_min": item.odds_min,
                "odds_max": item.odds_max,
                "pl_prob": tickets.wide_probability(prob_by_number, all_numbers, item.i, item.j),
            }
            for item in odds.wide
        ]
        print(formatters.format_pair_table("ワイド(i-j)", rows), end="")
        return 0

    if args.pl_fukusho:
        rows = [
            {
                "number": number,
                "name": name_by_number.get(number, ""),
                "pl_prob": tickets.fukusho_probability(prob_by_number, all_numbers, number),
            }
            for number in all_numbers
        ]
        rows.sort(key=lambda r: float(r["pl_prob"]), reverse=True)
        print(formatters.format_fukusho_table("複勝(3着以内)", rows), end="")
        return 0

    if args.ev_all:
        _print_ev_all(odds, prob_by_number, all_numbers, name_by_number)
        return 0

    # デフォルト：市場勝率の出力 + EV全表示
    if args.format == "json":
        payload = [
            {
                "number": r.number,
                "name": r.name,
                "win_odds": r.win_odds,
                "q": r.q,
                "market_win_prob": r.market_win_prob,
            }
            for r in market_rows
        ]
        output = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.write:
            Path(args.write).write_text(output, encoding="utf-8")
        else:
            print(output, end="")
    else:
        print(formatters.format_market_table(market_rows), end="")
        print()
        _print_ev_all(odds, prob_by_number, all_numbers, name_by_number)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
