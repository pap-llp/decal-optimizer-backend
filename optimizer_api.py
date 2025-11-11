from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from ortools.sat.python import cp_model
import itertools
import time

app = FastAPI()

# allow your HTML (Notify / Netlify) to call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RollItem(BaseModel):
    width: int
    count: int


class InputData(BaseModel):
    target: int
    rolls: List[RollItem]


@app.post("/optimize")
def optimize(data: InputData):
    target = data.target
    rolls = data.rolls

    # expand to flat items
    base_items: List[int] = []
    for r in rolls:
        base_items.extend([r.width] * r.count)

    # 1) solve baseline exactly (cp-sat but simple)
    best_baseline = solve_cp_sat(base_items, target, time_limit_s=8)

    # 2) try improvements: add up to 2 extra rolls per existing width
    improvements = try_improvements_cp(base_items, rolls, target, best_baseline["total_waste"])

    return {
        "best": best_baseline,
        "improvements": improvements
    }


def solve_cp_sat(items: List[int], target: int, time_limit_s: int = 8) -> Dict:
    """
    Exact-ish bin packing with CP-SAT:
    - each item goes into exactly one bin
    - each bin <= target
    - minimize (num_bins * BIG + total_waste)
    This is bounded by time_limit_s so it returns something.
    """
    model = cp_model.CpModel()
    n = len(items)
    max_bins = n  # worst case: each item its own bin
    if n > 50:
      # fallback to fast greedy if too large for CP-SAT within 8 s
      bins = pack_ffd(items, target)
      return {"bins": bins, "total_waste": sum(target - sum(b) for b in bins), "status": "GREEDY"}

    big = 10000   # weight to prioritize fewer bins

    # x[i][b] = item i in bin b
    x = {}
    for i in range(n):
      for b in range(max_bins):
        x[(i, b)] = model.NewBoolVar(f"x_{i}_{b}")

    # y[b] = bin b used
    y = [model.NewBoolVar(f"y_{b}") for b in range(max_bins)]

    # each item in exactly one bin
    for i in range(n):
        model.Add(sum(x[(i, b)] for b in range(max_bins)) == 1)

    # capacity constraints
    for b in range(max_bins):
        model.Add(
            sum(items[i] * x[(i, b)] for i in range(n))
            <= target
        ).OnlyEnforceIf(y[b])
        # if bin not used, capacity 0
        model.Add(
            sum(items[i] * x[(i, b)] for i in range(n))
            == 0
        ).OnlyEnforceIf(y[b].Not())

    # compute total waste = sum(target - load_b) over used bins
    # load_b = sum(items[i] * x[i,b])
    load_b = []
    wastes = []
    for b in range(max_bins):
        lb = model.NewIntVar(0, target, f"load_{b}")
        model.Add(lb == sum(items[i] * x[(i, b)] for i in range(n)))
        load_b.append(lb)
        wb = model.NewIntVar(0, target, f"waste_{b}")
        # waste = target - load, but 0 if bin not used
        model.Add(wb == target - lb).OnlyEnforceIf(y[b])
        model.Add(wb == 0).OnlyEnforceIf(y[b].Not())
        wastes.append(wb)

    total_waste = model.NewIntVar(0, target * max_bins, "total_waste")
    model.Add(total_waste == sum(wastes))

    # objective: minimize bins first, then waste
    model.Minimize(big * sum(y) + total_waste)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 20.0
    solver.parameters.num_search_workers = 8  # use parallelism if available

    res = solver.Solve(model)

    used_bins = []
    if res == cp_model.OPTIMAL or res == cp_model.FEASIBLE:
        for b in range(max_bins):
            if solver.Value(y[b]) == 1:
                bin_items = []
                for i in range(n):
                    if solver.Value(x[(i, b)]) == 1:
                        bin_items.append(items[i])
                if bin_items:
                    used_bins.append(bin_items)

        waste_val = sum(target - sum(bn) for bn in used_bins)
        return {
            "bins": used_bins,
            "total_waste": waste_val,
            "status": "OK"
        }

    # fallback
    return {
        "bins": [items],
        "total_waste": max(0, target - sum(items)),
        "status": "FALLBACK"
    }


def try_improvements_cp(items: List[int], rolls: List[RollItem], target: int, baseline_waste: int):
    """
    Try adding small numbers of extra rolls from existing widths.
    We try:
      - +1 of each width
      - +2 of each width
      - combinations of 2 widths (+1 each)
    For each candidate we run CP-SAT again (8s total allowed).
    """
    start = time.time()
    scenarios = []
    widths = [r.width for r in rolls]

    # candidates like: {210:1}, {210:2}, {210:1,158:1}, ...
    candidates = []

    # single-width additions
    for w in widths:
        candidates.append({w: 1})
        candidates.append({w: 2})

    # two-width additions (+1 each)
    for w1, w2 in itertools.combinations(widths, 2):
        candidates.append({w1: 1, w2: 1})

    for add in candidates:
        if time.time() - start > 8.0:  # don't overrun render free
            break

        new_items = list(items)
        for w, c in add.items():
            new_items.extend([w] * c)

        sol = solve_cp_sat(new_items, target, time_limit_s=3)  # shorter for subproblems
        if sol["total_waste"] < baseline_waste:
            scenarios.append({
                "added": add,
                "bins": sol["bins"],
                "total_waste": sol["total_waste"]
            })

    scenarios.sort(key=lambda s: s["total_waste"])
    return scenarios[:3]
