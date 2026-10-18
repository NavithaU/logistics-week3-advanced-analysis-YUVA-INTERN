"""
Week 3: Advanced Data Analysis and Visualization in Logistics
Dataset is hypothetical/simulated for educational use.
Run from this folder: python analyze_logistics.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "logistics_week3_hypothetical.csv"
VIS = ROOT / "visualizations"
OUT = ROOT / "outputs"
VIS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    numeric_cols = ["distance_km", "shipment_volume_units", "delivery_time_hours", "transport_cost_inr"]

    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nDescriptive statistics:\n", df[numeric_cols].describe().round(2))
    print("\nCorrelation matrix:\n", df[numeric_cols].corr().round(3))

    # 1. Distribution of delivery time: histogram helps understand spread and skew.
    plt.figure(figsize=(8, 4.5))
    plt.hist(df["delivery_time_hours"].dropna(), bins=14, edgecolor="black")
    plt.title("Distribution of Delivery Time")
    plt.xlabel("Delivery time (hours)")
    plt.ylabel("Number of shipments")
    plt.tight_layout()
    plt.savefig(VIS / "delivery_time_distribution.png", dpi=160)
    plt.close()

    # 2. Relationship between distance and cost: scatter plot reveals association/outliers.
    plt.figure(figsize=(8, 4.5))
    plt.scatter(df["distance_km"], df["transport_cost_inr"], alpha=0.75)
    plt.title("Transportation Cost vs Distance")
    plt.xlabel("Distance (km)")
    plt.ylabel("Transportation cost (INR)")
    plt.tight_layout()
    plt.savefig(VIS / "cost_vs_distance.png", dpi=160)
    plt.close()

    # 3. Average delivery time by vehicle type: bar chart compares groups.
    avg_vehicle = df.groupby("vehicle_type")["delivery_time_hours"].mean().sort_values()
    plt.figure(figsize=(8, 4.5))
    plt.bar(avg_vehicle.index, avg_vehicle.values)
    plt.title("Average Delivery Time by Vehicle Type")
    plt.xlabel("Vehicle type")
    plt.ylabel("Average delivery time (hours)")
    plt.tight_layout()
    plt.savefig(VIS / "avg_delivery_time_by_vehicle.png", dpi=160)
    plt.close()

    # 4. Average cost by region: bar chart compares regional cost levels.
    avg_region = df.groupby("region")["transport_cost_inr"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 4.5))
    plt.bar(avg_region.index, avg_region.values)
    plt.title("Average Transportation Cost by Region")
    plt.xlabel("Region")
    plt.ylabel("Average transportation cost (INR)")
    plt.tight_layout()
    plt.savefig(VIS / "avg_cost_by_region.png", dpi=160)
    plt.close()

    # 5. Correlation matrix: summarizes pairwise linear relationships.
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(7, 5))
    plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=25, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.colorbar(label="Correlation coefficient")
    plt.title("Correlation Matrix of Logistics Metrics")
    plt.tight_layout()
    plt.savefig(VIS / "correlation_matrix.png", dpi=160)
    plt.close()

    # 6. Daily trend and rolling average: helps see changes over time.
    daily = df.groupby("order_date")["delivery_time_hours"].mean().sort_index()
    rolling = daily.rolling(7, min_periods=1).mean()
    plt.figure(figsize=(9, 4.5))
    plt.plot(daily.index, daily.values, alpha=0.45, label="Daily average")
    plt.plot(rolling.index, rolling.values, label="7-day rolling average")
    plt.title("Delivery Time Trend Over Time")
    plt.xlabel("Order date")
    plt.ylabel("Delivery time (hours)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(VIS / "delivery_time_trend.png", dpi=160)
    plt.close()

    # Save tables for the report and further analysis.
    df[numeric_cols].describe().round(2).to_csv(OUT / "descriptive_statistics.csv")
    corr.round(3).to_csv(OUT / "correlation_matrix.csv")
    df.groupby("vehicle_type").agg(
        shipments=("order_id", "count"),
        avg_delivery_time_hours=("delivery_time_hours", "mean"),
        avg_transport_cost_inr=("transport_cost_inr", "mean"),
        avg_distance_km=("distance_km", "mean")
    ).round(2).to_csv(OUT / "vehicle_performance_summary.csv")
    df.groupby("region").agg(
        shipments=("order_id", "count"),
        avg_transport_cost_inr=("transport_cost_inr", "mean"),
        avg_delivery_time_hours=("delivery_time_hours", "mean")
    ).round(2).to_csv(OUT / "region_performance_summary.csv")

    print("\nDone. Charts saved in 'visualizations' and summary tables in 'outputs'.")

if __name__ == "__main__":
    main()
