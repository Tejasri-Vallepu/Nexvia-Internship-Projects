
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp

INPUT = "outputs/forecast_with_intervals.csv"
OUTPUT = "outputs/inventory_recommendations.csv"

df = pd.read_csv(INPUT)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

planning = (
    df.groupby("id", group_keys=False)
    .head(7)
    .groupby(
        ["id", "item_id", "store_id", "state_id"],
        as_index=False
    )
    .agg(
        forecast_demand=("lightgbm_prediction", "sum"),
        lower_demand=("lower_90", "sum"),
        upper_demand=("upper_90", "sum")
    )
)

# Demonstration assumptions
INITIAL_STOCK = 5
HOLDING_COST = 1.0
STOCKOUT_COST = 3.0
ORDER_COST = 0.2

# Scenario probabilities (illustrative)
PROBABILITIES = [0.2, 0.6, 0.2]


def optimize_inventory(base, lower, upper):

    solver = pywraplp.Solver.CreateSolver("SCIP")

    if solver is None:
        raise RuntimeError("SCIP solver unavailable")

    order = solver.IntVar(0, 100000, "order")

    demands = [
        max(0, int(np.ceil(lower))),
        max(0, int(np.ceil(base))),
        max(0, int(np.ceil(upper)))
    ]

    shortages = []
    excesses = []
    objective = solver.Objective()

    for i, (demand, probability) in enumerate(
        zip(demands, PROBABILITIES)
    ):
        shortage = solver.IntVar(
            0, 100000, f"shortage_{i}"
        )

        excess = solver.IntVar(
            0, 100000, f"excess_{i}"
        )

        solver.Add(
            INITIAL_STOCK + order - demand
            == excess - shortage
        )

        shortages.append(shortage)
        excesses.append(excess)

        objective.SetCoefficient(
            shortage,
            probability * STOCKOUT_COST
        )

        objective.SetCoefficient(
            excess,
            probability * HOLDING_COST
        )

    objective.SetCoefficient(order, ORDER_COST)
    objective.SetMinimization()

    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("Optimization did not reach optimum")

    expected_shortage = sum(
        p * s.solution_value()
        for p, s in zip(PROBABILITIES, shortages)
    )

    expected_excess = sum(
        p * e.solution_value()
        for p, e in zip(PROBABILITIES, excesses)
    )

    return {
        "order_quantity": int(round(order.solution_value())),
        "expected_shortage": round(expected_shortage, 2),
        "expected_excess": round(expected_excess, 2)
    }


print("Running scenario-based OR-Tools optimization...")

results = []

for row in planning.itertuples(index=False):

    solution = optimize_inventory(
        row.forecast_demand,
        row.lower_demand,
        row.upper_demand
    )

    results.append({
        "item_id": row.item_id,
        "store_id": row.store_id,
        "state_id": row.state_id,
        "forecast_7_day_demand": round(
            row.forecast_demand, 2
        ),
        "lower_demand": round(row.lower_demand, 2),
        "upper_demand": round(row.upper_demand, 2),
        "assumed_initial_stock": INITIAL_STOCK,
        **solution
    })

output = pd.DataFrame(results)

output.to_csv(OUTPUT, index=False)

print("\n" + "=" * 55)
print("SCENARIO-BASED INVENTORY OPTIMIZATION")
print("=" * 55)

print("Recommendations:", len(output))
print("Total order quantity:", output["order_quantity"].sum())
print("Expected shortage:",
      output["expected_shortage"].sum())
print("Expected excess:",
      output["expected_excess"].sum())

print("\nSample recommendations:")
print(output.head(10).to_string(index=False))

print("\nSaved:", OUTPUT)
