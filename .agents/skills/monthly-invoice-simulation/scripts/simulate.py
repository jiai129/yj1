"""Generate virtual invoice detail using quantity-weighted product sampling."""

import argparse
from collections import defaultdict
from decimal import Decimal, DecimalException
from heapq import heappop, heappush
import json
from math import gcd, isfinite
from pathlib import Path
from random import Random
from secrets import randbits


def positive_decimal(value):
    number = Decimal(str(value))
    if not number.is_finite() or number <= 0:
        raise ValueError(f"必须是正数：{value}")
    return number


def cents(value, *, allow_zero=False):
    number = Decimal(str(value))
    if not number.is_finite() or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"金额超出适用范围：{value}")
    scaled = number * 100
    if scaled != scaled.to_integral_value():
        raise ValueError(f"金额须精确到分：{value}")
    return int(scaled)


def money(value):
    sign = "-" if value < 0 else ""
    whole, fraction = divmod(abs(value), 100)
    return f"{sign}{whole}.{fraction:02d}"


def prepare_catalog(rows):
    references = defaultdict(lambda: defaultdict(Decimal))
    for row in rows:
        key = (row["name"].strip(), row["unit"].strip())
        if not all(key):
            raise ValueError("商品名称和单位不能为空")
        price = cents(row["unit_price"])
        references[key][price] += positive_decimal(row["quantity"])

    catalog = []
    for (name, unit), prices in references.items():
        weight = sum(prices.values())
        if not isfinite(float(weight)) or float(weight) == 0:
            raise ValueError(f"商品数量超出抽样范围：{name}")
        catalog.append((name, unit, weight, prices))
    if not catalog:
        raise ValueError("商品历史明细为空")
    return catalog


def draw(catalog, count, rng):
    pool = list(catalog)
    sample = []
    for _ in range(count):
        index = rng.choices(range(len(pool)), weights=[float(row[2]) for row in pool])[0]
        name, unit, weight, prices = pool.pop(index)
        price = rng.choices(list(prices), weights=[float(qty) for qty in prices.values()])[0]
        sample.append((name, unit, weight, price))
    return sample


def closest_quantities(prices, target, tolerance):
    # Each sampled product starts with one unit. Search additional quantities.
    minimum = sum(prices)
    upper = target + tolerance - minimum
    if upper < 0:
        return None
    divisor = gcd(*prices)
    steps = [price // divisor for price in prices]
    cheapest = min(range(len(steps)), key=steps.__getitem__)
    period = steps[cheapest]
    lower = max(0, (target - tolerance - minimum + divisor - 1) // divisor)
    limit = upper // divisor

    # The cheapest item can extend any reachable residue by one period at a time.
    # Find only the smallest reachable sum per residue, independent of the target size.
    smallest = {0: 0}
    predecessor = {}
    queue = [(0, 0)]
    while queue:
        distance, residue = heappop(queue)
        if distance != smallest[residue]:
            continue
        for index, step in enumerate(steps):
            new_distance = distance + step
            new_residue = new_distance % period
            if new_distance <= limit and new_distance < smallest.get(new_residue, limit + 1):
                smallest[new_residue] = new_distance
                predecessor[new_residue] = (residue, index)
                heappush(queue, (new_distance, new_residue))

    desired = (target - minimum) // divisor
    best = None
    for residue, first in smallest.items():
        first_repeat = max(0, (lower - first + period - 1) // period)
        last_repeat = (limit - first) // period
        if first_repeat > last_repeat:
            continue
        nearest_repeat = (desired - first) // period
        for repeat in (nearest_repeat, nearest_repeat + 1):
            repeat = min(last_repeat, max(first_repeat, repeat))
            total = minimum + divisor * (first + repeat * period)
            candidate = (abs(total - target), total, residue, repeat)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        return None
    _, total, residue, repeats = best
    quantities = [1] * len(prices)
    quantities[cheapest] += repeats
    while residue:
        residue, index = predecessor[residue]
        quantities[index] += 1
    return total, quantities


def simulate(catalog, target, tolerance, item_counts, attempts, rng):
    best = None
    for _ in range(attempts):
        item_count = rng.choice(item_counts) if len(item_counts) > 1 else item_counts[0]
        sample = draw(catalog, item_count, rng)
        match = closest_quantities([item[3] for item in sample], target, tolerance)
        if match is None:
            continue
        total, quantities = match
        if best is None or abs(total - target) < abs(best[0] - target):
            best = (total, quantities, sample)
        if total == target:
            break
    if best is None:
        return {"status": "unmatched", "total": None, "difference": None, "items": []}

    total, quantities, sample = best
    items = []
    for (name, unit, weight, price), quantity in zip(sample, quantities):
        items.append({
            "name": name,
            "unit": unit,
            "historical_quantity": format(weight, "f"),
            "reference_unit_price": money(price),
            "virtual_quantity": quantity,
            "amount": money(price * quantity),
        })
    return {
        "status": "exact" if total == target else "within_tolerance",
        "total": money(total),
        "difference": money(total - target),
        "items": items,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, help="UTF-8 JSON with products and targets")
    parser.add_argument("--output", type=Path, help="Save JSON to this path instead of stdout")
    parser.add_argument("--items", type=int, help="Fixed product count; default varies from 1 to 5")
    parser.add_argument("--keep-cents", action="store_true", help="Keep target cents instead of truncating to whole yuan")
    parser.add_argument("--tolerance", default="10.00", help="Order-total tolerance in yuan")
    parser.add_argument("--attempts", type=int, default=100, help="Maximum draws per target")
    parser.add_argument("--seed", type=int, help="Reuse a seed to reproduce a run")
    args = parser.parse_args()
    try:
        batch = json.loads(args.batch.read_text(encoding="utf-8-sig"), parse_float=Decimal)
        catalog = prepare_catalog(batch["products"])
        if (args.items is not None and not 1 <= args.items <= len(catalog)) or args.attempts < 1:
            raise ValueError("抽样商品种数须在商品库范围内，抽样次数须为正数")
        item_counts = (args.items,) if args.items is not None else tuple(range(1, min(5, len(catalog)) + 1))
        tolerance = cents(args.tolerance, allow_zero=True)
        seed = randbits(64) if args.seed is None else args.seed
        rng = Random(seed)
        results = []
        ids = set()
        for row in batch["targets"]:
            store = row["id"].strip()
            if not store or store in ids:
                raise ValueError("门店 ID 须非空且唯一")
            ids.add(store)
            source_target = positive_decimal(row["amount"])
            target = cents(source_target) if args.keep_cents else int(source_target) * 100
            if target:
                result = simulate(catalog, target, tolerance, item_counts, args.attempts, rng)
            else:
                result = {"status": "unmatched", "total": None, "difference": None,
                          "items": [], "reason": "取整后金额为零"}
            results.append({"id": store, "source_target": format(source_target, "f"),
                            "target": money(target), **result})
        output = json.dumps({"test_only": True, "seed": seed, "results": results}, ensure_ascii=False, indent=2) + "\n"
        if args.output is None:
            print(output, end="")
        else:
            args.output.write_text(output, encoding="utf-8")
    except (AttributeError, DecimalException, KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
