from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import itertools
import time

app = FastAPI()

# CORS so your HTML (on Notify/Netlify) can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later to your domain
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

    # Expand to flat list of widths
    items: List[int] = []
    for r in rolls:
      items.extend([r.width] * r.count)

    # 1) Baseline greedy pack
    baseline_bins = pack_ffd(items, target)
    baseline_waste = total_waste(baseline_bins, target)

    # 2) Bounded exhaustive improvement search
    improvements = search_improvements(items, rolls, target, baseline_waste)

    return {
        "best": {
            "bins": baseline_bins,
            "total_waste": baseline_waste
        },
        "improvements": improvements
    }


def pack_ffd(items: List[int], target: int) -> List[List[int]]:
    """
    First-fit decreasing packing — fast, deterministic.
    """
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


def search_improvements(items: List[int], rolls: List[RollItem], target: int, baseline_waste: int):
    """
    Try adding extra rolls ONLY from existing widths.
    Bounded exhaustive: each width can add 0..3, total added <= 6.
    This is to catch cases like “add 1 × 210mm → lower waste”.
    """
    widths = [r.width for r in rolls]
    start = time.time()
    scenarios = []

    # for each width we allow adding 0,1,2,3 — then we prune combos whose total added > 6
    per_width_choices = {w: [0, 1, 2, 3] for w in widths}

    def all_addition_maps():
        choice_lists = [per_width_choices[w] for w in widths]
        for tup in itertools.product(*choice_lists):
            total_added = sum(tup)
            if 0 < total_added <= 6:
                yield {w: c for w, c in zip(widths, tup) if c > 0}

    for add_map in all_addition_maps():
        # time guard (Render free tier)
        if time.time() - start > 8.0:
            break

        # build new list of items
        new_items = list(items)
        for w, c in add_map.items():
            new_items.extend([w] * c)

        # repack
        packed = pack_ffd(new_items, target)
        waste = total_waste(packed, target)

        if waste < baseline_waste:
            scenarios.append({
                "added": add_map,
                "bins": packed,
                "total_waste": waste
            })

    # sort by best waste and return top 3
    scenarios.sort(key=lambda s: s["total_waste"])
    return scenarios[:3]
