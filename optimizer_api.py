from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from ortools.linear_solver import pywraplp

app = FastAPI()

# CORS FIX
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace * with your frontend URL for production
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

    # Expand rolls into individual items
    items = []
    for r in rolls:
        items.extend([r.width] * r.count)

    best_plan = solve_optimal(items, target)
    improvements = find_improvements(rolls, target, best_plan)

    return {"best": best_plan, "improvements": improvements}


def solve_optimal(items, target, max_time_sec=5):
    """Integer programming to minimize total waste (with time limit)."""
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        return {"bins": [], "total_waste": None, "error": "Solver not available"}

    n = len(items)
    max_bins = n

    # Variables
    x = [[solver.BoolVar(f"x_{i}_{b}") for b in range(max_bins)] for i in range(n)]
    y = [solver.BoolVar(f"y_{b}") for b in range(max_bins)]

    # Constraints
    for i in range(n):
        solver.Add(sum(x[i][b] for b in range(max_bins)) == 1)
    for b in range(max_bins):
        solver.Add(sum(items[i] * x[i][b] for i in range(n)) <= target * y[b])

    # Objective: minimize both number of bins and total waste
    total_waste_expr = solver.Sum(target * y[b] - solver.Sum(items[i] * x[i][b] for i in range(n)) for b in range(max_bins))
    solver.Minimize(1000 * solver.Sum(y[b] for b in range(max_bins)) + total_waste_expr)


    # Limit runtime to 5 seconds
    solver.SetTimeLimit(max_time_sec * 1000)

    status = solver.Solve()
    bins, used_bins = [], []

    for b in range(max_bins):
        bin_items = [items[i] for i in range(n) if x[i][b].solution_value() > 0.5]
        if bin_items:
            bins.append(bin_items)
            used_bins.append(bin_items)

    total_waste = sum(target - sum(b) for b in used_bins)
    return {
        "bins": used_bins,
        "total_waste": total_waste,
        "status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "APPROXIMATE"
    }



def find_improvements(rolls, target, best_plan, max_extra=5):
    """Explore multiple extra rolls (1–5) for each existing width."""
    improvements = []

    for r in rolls:
        for extra in range(1, max_extra + 1):
            test_rolls = [RollItem(width=t.width, count=t.count) for t in rolls]
            for t in test_rolls:
                if t.width == r.width:
                    t.count += extra

            items = []
            for t in test_rolls:
                items.extend([t.width] * t.count)

            new_plan = solve_optimal(items, target)
            if new_plan["total_waste"] < best_plan["total_waste"]:
                improvements.append({
                    "added_width": r.width,
                    "extra_rolls": extra,
                    "total_waste": int(new_plan["total_waste"]),
                    "bins": new_plan["bins"]
                })

    # Sort by total waste ascending
    improvements.sort(key=lambda x: x["total_waste"])
    return improvements[:3]

