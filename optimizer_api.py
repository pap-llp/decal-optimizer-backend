from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from ortools.sat.python import cp_model
from concurrent.futures import ThreadPoolExecutor, TimeoutError as TOut
import itertools
import time
import asyncio

# ======================================================
# FastAPI setup
# ======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# Models
# ======================================================
class RollItem(BaseModel):
    width: int
    count: int

class InputData(BaseModel):
    target: int
    rolls: List[RollItem]

# ======================================================
# Helpers
# ======================================================
def pack_ffd(items: List[int], target: int) -> List[List[int]]:
    """Fast greedy: first-fit decreasing."""
    arr = sorted(items, reverse=True)
    bins: List[List[int]] = []
    for w in arr:
        placed = False
        for b in bins:
            if sum(b) + w <= target:
                b.append(w)
                placed = True
                break
        if not placed:
            bins.append([w])
    return bins

def total_waste(bins: List[List[int]], target: int) -> int:
    return sum(target - sum(b) for b in bins)

# ======================================================
# CP-SAT core
# ======================================================
def solve_cp_sat(items: List[int], target: int, time_limit_s: int = 12) -> Dict[str, Any]:
    """Exact-ish bin packing with CP-SAT, time-bounded."""
    print(f"[solver] CP-SAT start | items={len(items)} | limit={time_limit_s}s")
    model = cp_model.CpModel()
    n = len(items)
    max_bins = n  # worst case
    big = 10000   # weight for bin count

    # x[i,b] = 1 if item i goes to bin b
    x = {}
    for i in range(n):
        for b in range(max_bins):
            x[(i, b)] = model.NewBoolVar(f"x_{i}_{b}")

    # y[b] = 1 if bin b is used
    y = [model.NewBoolVar(f"y_{b}") for b in range(max_bins)]

    # each item exactly once
    for i in range(n):
        model.Add(sum(x[(i, b)] for b in range(max_bins)) == 1)

    # capacity and waste per bin
    load_b = []
    waste_b = []
    for b in range(max_bins):
        lb = model.NewIntVar(0, target, f"load_{b}")
        model.Add(lb == sum(items[i] * x[(i, b)] for i in range(n)))
        load_b.append(lb)

        # if used -> load <= target, else load == 0
        model.Add(lb <= target).OnlyEnforceIf(y[b])
        model.Add(lb == 0).OnlyEnforceIf(y[b].Not())

        wb = model.NewIntVar(0, target, f"waste_{b}")
        model.Add(wb == target - lb).OnlyEnforceIf(y[b])
        model.Add(wb == 0).OnlyEnforceIf(y[b].Not())
        waste_b.append(wb)

    total_waste_v = model.NewIntVar(0, target * max_bins, "total_waste")
    model.Add(total_waste_v == sum(waste_b))

    # objective: fewer bins, then less waste
    model.Minimize(big * sum(y) + total_waste_v)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)

    used_bins: List[List[int]] = []
    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for b in range(max_bins):
            if solver.Value(y[b]) == 1:
                bin_items = [items[i] for i in range(n) if solver.Value(x[(i, b)]) == 1]
                if bin_items:
                    used_bins.append(bin_items)
        waste_val = sum(target - sum(bn) for bn in used_bins)
        print(f"[solver] CP-SAT done | bins={len(used_bins)} | waste={waste_val}")
        return {"bins": used_bins, "total_waste": waste_val, "status": "OK"}

    # fallback
    print("[solver] CP-SAT failed, fallback greedy")
    bins = pack_ffd(items, target)
    return {
        "bins": bins,
        "total_waste": total_waste(bins, target),
        "status": "FALLBACK"
    }

# ======================================================
# Threaded guard (so we always return)
# ======================================================
executor = ThreadPoolExecutor(max_workers=1)

def solve_with_timeout(items: List[int], target: int, limit: int = 12) -> Dict[str, Any]:
    fut = executor.submit(solve_cp_sat, items, target, limit)
    try:
        return fut.result(timeout=limit + 2)
    except TOut:
        print("[solver] TIMEOUT — fallback greedy")
        bins = pack_ffd(items, target)
        return {
            "bins": bins,
            "total_waste": total_waste(bins, target),
            "status": "TIMEOUT"
        }

# ======================================================
# Deep search (multi-stage)
# ======================================================
def deep_search_improvements(base_items: List[int],
                             rolls: List[RollItem],
                             target: int,
                             baseline_waste: int,
                             stages: int = 3,
                             per_stage_limit_s: int = 12) -> List[Dict[str, Any]]:
    """
    Multi-stage improvement:
    - Stage 1: try small additions
    - Stage 2: take best found, try again
    - Stage 3: repeat
    Keep top 3 overall.
    """
    print(f"[deep] starting deep search | baseline={baseline_waste}")
    widths = [r.width for r in rolls]
    best_overall: List[Dict[str, Any]] = []

    # current best items to improve upon
    current_best_items = base_items[:]
    current_best_waste = baseline_waste

    for stage in range(1, stages + 1):
        print(f"[deep] stage {stage} start")
        stage_candidates = generate_addition_candidates(widths, max_extra_per_width=2, max_total_extra=6)
        stage_best: List[Dict[str, Any]] = []

        stage_start = time.time()
        for add_map in stage_candidates:
            # time guard per stage
            if time.time() - stage_start > per_stage_limit_s:
                print(f"[deep] stage {stage} time limit hit")
                break

            # build new list with additions
            new_items = current_best_items[:]
            for w, c in add_map.items():
                new_items.extend([w] * c)

            # solve (bounded)
            sol = solve_with_timeout(new_items, target, limit=per_stage_limit_s)

            if sol["total_waste"] < current_best_waste:
                improved = {
                    "added": add_map,
                    "bins": sol["bins"],
                    "total_waste": sol["total_waste"],
                    "stage": stage
                }
                stage_best.append(improved)
                best_overall.append(improved)
                # update current baseline for next stage
                current_best_items = new_items
                current_best_waste = sol["total_waste"]
                print(f"[deep] stage {stage} improvement -> waste={sol['total_waste']} add={add_map}")

        if not stage_best:
            print(f"[deep] stage {stage} no improvements")

    # sort all improvements found across stages
    best_overall.sort(key=lambda x: x["total_waste"])
    top3 = best_overall[:3]
    print(f"[deep] done | found={len(best_overall)} | returning={len(top3)}")
    return top3


def generate_addition_candidates(widths: List[int],
                                 max_extra_per_width: int = 2,
                                 max_total_extra: int = 6):
    """
    Generate maps like {210:1}, {210:2}, {210:1, 174:1}, ...
    but keep total added rolls <= max_total_extra
    """
    # per-width choices: 0..max_extra_per_width
    choices_per_width = [[i for i in range(max_extra_per_width + 1)] for _ in widths]
    for tuple_counts in itertools.product(*choices_per_width):
        total_added = sum(tuple_counts)
        if 0 < total_added <= max_total_extra:
            add_map = {}
            for w, c in zip(widths, tuple_counts):
                if c > 0:
                    add_map[w] = c
            yield add_map

# ======================================================
# API endpoint
# ======================================================
@app.post("/optimize")
async def optimize(data: InputData):
    await asyncio.sleep(0)  # keep event loop responsive
    target = data.target
    rolls = data.rolls

    # expand to flat list
    items: List[int] = []
    for r in rolls:
        items.extend([r.width] * r.count)

    # big inputs: fallback immediately (Render free safety)
    if len(items) > 80:
        print("[api] large input -> greedy only")
        bins = pack_ffd(items, target)
        tw = total_waste(bins, target)
        return {
            "best": {"bins": bins, "total_waste": tw, "status": "GREEDY_LARGE"},
            "improvements": []
        }

    # baseline solve
    baseline = solve_with_timeout(items, target, limit=12)
    print(f"[api] baseline waste={baseline['total_waste']}")

    # deep search for improvements
    improvements = deep_search_improvements(
        base_items=items,
        rolls=rolls,
        target=target,
        baseline_waste=baseline["total_waste"],
        stages=3,
        per_stage_limit_s=12
    )

    return {
        "best": baseline,
        "improvements": improvements
    }
