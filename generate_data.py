import numpy as np
import pandas as pd
from faker import Faker
from pathlib import Path

fake = Faker()
rng = np.random.default_rng(42)

def month_start(dt: pd.Series) -> pd.Series:
    return dt.dt.to_period("M").dt.to_timestamp()

def months_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)

def main():
    out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_customers = 3000
    n_orders = 45000

    start_date = pd.Timestamp("2016-09-04")
    end_date = pd.Timestamp("2018-10-17")

    customer_ids = np.arange(1, n_customers + 1)

    orders = pd.DataFrame({
        "order_id": np.arange(1, n_orders + 1),
        "customer_id": rng.choice(customer_ids, size=n_orders, replace=True),
    })

    total_seconds = int((end_date - start_date).total_seconds())
    random_seconds = rng.integers(0, total_seconds + 1, size=n_orders)
    orders["order_purchase_ts"] = start_date + pd.to_timedelta(random_seconds, unit="s")
    orders["order_date"] = orders["order_purchase_ts"].dt.date
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    status_choices = np.array(["Delivered", "Canceled", "Processing"])
    status_probs = np.array([0.88, 0.07, 0.05])
    orders["order_status"] = rng.choice(status_choices, size=n_orders, p=status_probs)

    base_item = rng.gamma(shape=2.2, scale=35.0, size=n_orders)
    item_value = np.clip(base_item, 8, 800).round(2)

    freight_value = rng.normal(loc=12.0, scale=6.0, size=n_orders)
    freight_value = np.clip(freight_value, 0, 60).round(2)

    gross_value = (item_value + freight_value).round(2)

    delivery_days = rng.normal(loc=7.5, scale=3.0, size=n_orders)
    delivery_days = np.clip(delivery_days, 1, 30).round(0).astype(int)

    estimated_days = rng.normal(loc=8.0, scale=2.5, size=n_orders)
    estimated_days = np.clip(estimated_days, 2, 30).round(0).astype(int)

    delivered_minus_estimated = (delivery_days - estimated_days).astype(int)

    orders["item_value"] = item_value
    orders["freight_value"] = freight_value
    orders["gross_value"] = gross_value
    orders["delivery_days"] = delivery_days
    orders["delivered_minus_estimated_days"] = delivered_minus_estimated
    orders["on_time_delivery_flag"] = (orders["delivered_minus_estimated_days"] <= 0).astype(int)

    orders.loc[orders["order_status"] != "Delivered", "on_time_delivery_flag"] = np.nan

    fact_orders_path = out_dir / "fact_orders.csv"
    orders.to_csv(fact_orders_path, index=False)

    delivered = orders[orders["order_status"] == "Delivered"].copy()
    delivered["order_month"] = month_start(delivered["order_date"])

    cohort = (
        delivered.groupby("customer_id", as_index=False)["order_month"]
        .min()
        .rename(columns={"order_month": "cohort_month"})
    )

    d2 = delivered.merge(cohort, on="customer_id", how="left")
    d2["cohort_age_months"] = months_between(d2["cohort_month"], d2["order_month"])

    cohort_customers = (
        d2.groupby("cohort_month", as_index=False)["customer_id"]
        .nunique()
        .rename(columns={"customer_id": "cohort_customers"})
    )

    active = (
        d2.groupby(["cohort_month", "order_month", "cohort_age_months"], as_index=False)["customer_id"]
        .nunique()
        .rename(columns={"customer_id": "active_customers"})
    )

    metrics = active.merge(cohort_customers, on="cohort_month", how="left")
    metrics["retention_rate"] = (metrics["active_customers"] / metrics["cohort_customers"]).round(6)

    metrics = metrics.sort_values(["cohort_month", "cohort_age_months"]).reset_index(drop=True)

    metrics_path = out_dir / "metrics_cohort_retention_monthly.csv"
    metrics.to_csv(metrics_path, index=False)

    print("Saved files:")
    print(f"  {fact_orders_path.resolve()}")
    print(f"  {metrics_path.resolve()}")
    print()
    print("Rows:")
    print(f"  fact_orders: {len(orders)}")
    print(f"  metrics_cohort_retention_monthly: {len(metrics)}")

if __name__ == "__main__":
    main()
