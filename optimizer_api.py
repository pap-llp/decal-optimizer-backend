from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from ortools.sat.python import cp_model
from concurrent.futures import ThreadPoolExecutor, TimeoutError as TOut
import itertools, asyncio

# -------------------- App setup --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# -------------------- Greedy fallback --------------------
def pack_ffd(items: List[int], target: int) -> List[List[int]]:
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

# -------------------- CP-SAT solver --------------------
def solve_cp_sat(items: List[int], target: int, time_limit_s: int = 15) -> Dict[str, Any]:
    model = cp_model.CpModel()
    n = len(items)
    max_bins = n
    big = 10000

    x = {(i, b): model.NewBoolVar(f"x_{i}_{b}") for i in range(n) for b in range(max_bins)}
    y = [model.NewBoolVar(f"y_{b}") for b in range(max_bins)]

    for i in range(n):
        model.Add(sum(x[(i, b)] for b in range(max_bins)) == 1)

    load_b, waste_b = [], []
    for b in range(max_bins):
        lb = model.NewIntVar(0, target, f"load_{b}")
        model.Add(lb == sum(items[i] * x[(i, b)] for i in range(n)))
        load_b.append(lb)
        model.Add(lb <= target).OnlyEnforceIf(y[b])
        model.Add(lb == 0).OnlyEnforceIf(y[b].Not())
        wb = model.NewIntVar(0, target, f"waste_{b}")
        model.Add(wb == target - lb).OnlyEnforceIf(y[b])
        model.Add(wb == 0).OnlyEnforceIf(y[b].Not())
        waste_b.append(wb)

    total_waste_v = model.NewIntVar(0, target * max_bins, "total_waste")
    model.Add(total_waste_v == sum(waste_b))
    model.Minimize(big * sum(y) + total_waste_v)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    res = solver.Solve(model)

    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        used_bins: List[List[int]] = []
        for b in range(max_bins):
            if solver.Value(y[b]) == 1:
                bin_items = [items[i] for i in range(n) if solver.Value(x[(i, b)]) == 1]
                if bin_items:
                    used_bins.append(bin_items)
        return {"bins": used_bins, "total_waste": total_waste(used_bins, target), "status": "OK"}

    bins = pack_ffd(items, target)
    return {"bins": bins, "total_waste": total_waste(bins, target), "status": "FALLBACK"}

executor = ThreadPoolExecutor(max_workers=1)
def solve_with_timeout(items, target, limit=15):
    fut = executor.submit(solve_cp_sat, items, target, limit)
    try:
        return fut.result(timeout=limit + 3)
    except TOut:
        bins = pack_ffd(items, target)
        return {"bins": bins, "total_waste": total_waste(bins, target), "status": "TIMEOUT"}

# -------------------- Global improvement search --------------------
def global_improvement_search(base_items: List[int],
                              rolls: List[RollItem],
                              target: int,
                              baseline_waste: int,
                              max_add: int = 3,
                              max_remove: int = 2,
                              limit_per_eval: int = 15,
                              top_k: int = 3):
    widths = [r.width for r in rolls]
    improvements: List[Dict[str, Any]] = []
    seen = set()

    add_choices = [range(max_add + 1) for _ in widths]
    rem_choices = [range(max_remove + 1) for _ in widths]

    for add_tuple in itertools.product(*add_choices):
        for rem_tuple in itertools.product(*rem_choices):
            total_add = sum(add_tuple)
            total_rem = sum(rem_tuple)
            if total_add == 0 and total_rem == 0:
                continue
            if total_add > 3 or total_rem > 2:
                continue
            key = (tuple(add_tuple), tuple(rem_tuple))
            if key in seen:
                continue
            seen.add(key)

            new_items = base_items[:]
            # remove
            for w, rcount in zip(widths, rem_tuple):
                for _ in range(rcount):
                    if w in new_items:
                        new_items.remove(w)
            # add
            for w, acount in zip(widths, add_tuple):
                new_items.extend([w] * acount)

            sol = solve_with_timeout(new_items, target, limit_per_eval)
            if sol["total_waste"] < baseline_waste:
                improvements.append({
                    "added": {w: a for w, a in zip(widths, add_tuple) if a > 0},
                    "removed": {w: r for w, r in zip(widths, rem_tuple) if r > 0},
                    "bins": sol["bins"],
                    "total_waste": sol["total_waste"],
                    "status": sol["status"]
                })
    improvements.sort(key=lambda x: x["total_waste"])
    return improvements[:top_k]

# -------------------- API --------------------
@app.post("/optimize")
async def optimize(data: InputData):
    await asyncio.sleep(0)
    target = data.target
    rolls = data.rolls
    items = []
    for r in rolls:
        items.extend([r.width] * r.count)

    if len(items) > 80:
        bins = pack_ffd(items, target)
        tw = total_waste(bins, target)
        return {"best": {"bins": bins, "total_waste": tw, "status": "GREEDY_LARGE"}, "improvements": []}

    baseline = solve_with_timeout(items, target, limit=15)
    improvements = global_improvement_search(items, rolls, target, baseline["total_waste"])
    return {"best": baseline, "improvements": improvements}
