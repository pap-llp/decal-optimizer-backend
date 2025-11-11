from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from ortools.sat.python import cp_model
from concurrent.futures import ThreadPoolExecutor, TimeoutError as TOut
import itertools, time, asyncio

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

# -------- Greedy fallback --------
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

# -------- CP-SAT core solver --------
def solve_cp_sat(items: List[int], target: int, time_limit_s: int = 8) -> Dict:
    model = cp_model.CpModel()
    n = len(items)
    max_bins = n
    big = 10000

    x = {}
    for i in range(n):
        for b in range(max_bins):
            x[(i,b)] = model.NewBoolVar(f"x_{i}_{b}")
    y = [model.NewBoolVar(f"y_{b}") for b in range(max_bins)]

    for i in range(n):
        model.Add(sum(x[(i,b)] for b in range(max_bins)) == 1)
    for b in range(max_bins):
        model.Add(sum(items[i]*x[(i,b)] for i in range(n)) <= target).OnlyEnforceIf(y[b])
        model.Add(sum(items[i]*x[(i,b)] for i in range(n)) == 0).OnlyEnforceIf(y[b].Not())

    load_b, wastes = [], []
    for b in range(max_bins):
        lb = model.NewIntVar(0, target, f"load_{b}")
        model.Add(lb == sum(items[i]*x[(i,b)] for i in range(n)))
        load_b.append(lb)
        wb = model.NewIntVar(0, target, f"waste_{b}")
        model.Add(wb == target - lb).OnlyEnforceIf(y[b])
        model.Add(wb == 0).OnlyEnforceIf(y[b].Not())
        wastes.append(wb)

    total_waste_v = model.NewIntVar(0, target*max_bins, "total_waste")
    model.Add(total_waste_v == sum(wastes))
    model.Minimize(big*sum(y) + total_waste_v)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8

    res = solver.Solve(model)
    used_bins = []
    if res in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for b in range(max_bins):
            if solver.Value(y[b]) == 1:
                bin_items = [items[i] for i in range(n) if solver.Value(x[(i,b)]) == 1]
                if bin_items: used_bins.append(bin_items)
        waste_val = sum(target - sum(bn) for bn in used_bins)
        return {"bins": used_bins, "total_waste": waste_val, "status": "OK"}
    return {"bins": [items], "total_waste": max(0, target - sum(items)), "status": "FALLBACK"}

# -------- Hybrid guarded solver --------
executor = ThreadPoolExecutor(max_workers=1)

def solve_with_timeout(items, target, limit=8):
    fut = executor.submit(solve_cp_sat, items, target, limit)
    try:
        return fut.result(timeout=limit+2)
    except TOut:
        fut.cancel()
        bins = pack_ffd(items, target)
        return {"bins": bins, "total_waste": total_waste(bins, target), "status": "TIMEOUT"}

def try_improvements(items, rolls, target, baseline_waste):
    start = time.time()
    scenarios, widths = [], [r.width for r in rolls]
    candidates = []
    for w in widths:
        candidates.append({w: 1}); candidates.append({w: 2})
    for w1, w2 in itertools.combinations(widths, 2):
        candidates.append({w1: 1, w2: 1})
    for add in candidates:
        if time.time()-start > 8.0: break
        new_items = list(items)
        for w,c in add.items(): new_items.extend([w]*c)
        sol = solve_with_timeout(new_items, target, limit=3)
        if sol["total_waste"] < baseline_waste:
            scenarios.append({"added": add, "bins": sol["bins"], "total_waste": sol["total_waste"]})
    scenarios.sort(key=lambda s: s["total_waste"])
    return scenarios[:3]

# -------- API endpoint --------
@app.post("/optimize")
async def optimize(data: InputData):
    await asyncio.sleep(0)
    target = data.target
    rolls = data.rolls
    items = []
    for r in rolls: items.extend([r.width]*r.count)
    if len(items) > 50:
        # fallback early for huge sets
        bins = pack_ffd(items, target)
        total = total_waste(bins, target)
        return {"best": {"bins": bins, "total_waste": total, "status": "GREEDY"}, "improvements": []}
    best = solve_with_timeout(items, target, limit=8)
    improvements = try_improvements(items, rolls, target, best["total_waste"])
    return {"best": best, "improvements": improvements}
