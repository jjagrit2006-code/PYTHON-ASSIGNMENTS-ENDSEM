import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

# -----------------------------------------------------
# LOGGING CONFIGURATION
# -----------------------------------------------------
logging.basicConfig(
    filename="energy_dashboard.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -----------------------------------------------------
# TASK 1 — DATA INGESTION & VALIDATION
# -----------------------------------------------------

def load_energy_data(data_folder="data"):
    combined_df = pd.DataFrame()
    data_path = Path(data_folder)

    if not data_path.exists():
        logging.error("Data folder does not exist.")
        return pd.DataFrame()

    for file in data_path.glob("*.csv"):
        try:
            df = pd.read_csv(file, on_bad_lines="skip")
            df["building"] = file.stem  # filename as building name

            # Timestamp standardization
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                raise ValueError("timestamp column missing")

            if "kwh" not in df.columns:
                raise ValueError("kwh column missing")

            combined_df = pd.concat([combined_df, df], ignore_index=True)

        except FileNotFoundError:
            logging.error(f"File not found: {file}")
        except Exception as e:
            logging.error(f"Error processing {file}: {e}")

    return combined_df


# -----------------------------------------------------
# TASK 2 — AGGREGATION FUNCTIONS
# -----------------------------------------------------

def calculate_daily_totals(df):
    df = df.set_index("timestamp")
    return df.resample("D")["kwh"].sum().reset_index()

def calculate_weekly_aggregates(df):
    df = df.set_index("timestamp")
    return df.resample("W")["kwh"].sum().reset_index()

def building_wise_summary(df):
    summary = df.groupby("building")["kwh"].agg(
        total="sum",
        mean="mean",
        min="min",
        max="max"
    ).reset_index()
    return summary


# -----------------------------------------------------
# TASK 3 — OOP DESIGN
# -----------------------------------------------------

class MeterReading:
    def __init__(self, timestamp, kwh):
        self.timestamp = timestamp
        self.kwh = kwh

class Building:
    def __init__(self, name):
        self.name = name
        self.meter_readings = []

    def add_reading(self, timestamp, kwh):
        self.meter_readings.append(MeterReading(timestamp, kwh))

    def calculate_total_consumption(self):
        return sum(r.kwh for r in self.meter_readings)

    def generate_report(self):
        total = self.calculate_total_consumption()
        return f"Building: {self.name} — Total Consumption: {total:.2f} kWh"

class BuildingManager:
    def __init__(self):
        self.buildings = {}

    def add_reading(self, building_name, timestamp, kwh):
        if building_name not in self.buildings:
            self.buildings[building_name] = Building(building_name)
        self.buildings[building_name].add_reading(timestamp, kwh)

    def generate_all_reports(self):
        return [b.generate_report() for b in self.buildings.values()]


# -----------------------------------------------------
# TASK 4 — DASHBOARD VISUALIZATION
# -----------------------------------------------------

def create_dashboard(df):
    fig, axs = plt.subplots(3, 1, figsize=(12, 16))

    # 1 — Trend Line (Daily Consumption per Building)
    for b in df["building"].unique():
        sub = df[df["building"] == b].set_index("timestamp").resample("D")["kwh"].sum()
        axs[0].plot(sub.index, sub.values, label=b)
    axs[0].set_title("Daily Electricity Consumption")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("kWh")
    axs[0].legend()

    # 2 — Weekly Average Bar Chart
    weekly = df.set_index("timestamp").groupby("building")["kwh"].resample("W").sum().groupby("building").mean()
    axs[1].bar(weekly.index, weekly.values)
    axs[1].set_title("Average Weekly Usage by Building")
    axs[1].set_ylabel("kWh")

    # 3 — Scatter Plot — Peak Hour vs kWh
    axs[2].scatter(df["timestamp"], df["kwh"])
    axs[2].set_title("Peak Hour Consumption Scatter")
    axs[2].set_xlabel("Timestamp")
    axs[2].set_ylabel("kWh")

    plt.tight_layout()
    plt.savefig("dashboard.png")
    plt.close()


# -----------------------------------------------------
# TASK 5 — EXPORTS + SUMMARY
# -----------------------------------------------------

def generate_summary(df, summary_df):
    total_consumption = df["kwh"].sum()
    highest_building = summary_df.loc[summary_df["total"].idxmax()]["building"]

    peak_row = df.loc[df["kwh"].idxmax()]
    peak_time = peak_row["timestamp"]
    peak_load = peak_row["kwh"]

    daily = calculate_daily_totals(df)
    weekly = calculate_weekly_aggregates(df)

    summary_text = f"""
Campus Energy Use Summary
--------------------------------------
Total Campus Consumption: {total_consumption:.2f} kWh
Highest Consuming Building: {highest_building}
Peak Load Time: {peak_time} with {peak_load:.2f} kWh

Daily Trend Range: {daily['kwh'].min():.2f} — {daily['kwh'].max():.2f}
Weekly Trend Range: {weekly['kwh'].min():.2f} — {weekly['kwh'].max():.2f}
"""

    with open("summary.txt", "w") as f:
        f.write(summary_text)

    print(summary_text)


# -----------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------

def main():
    print("Loading data...")
    df = load_energy_data()

    if df.empty:
        print("No data found. Check /data folder.")
        return

    print("Processing summaries...")
    summary_df = building_wise_summary(df)

    summary_df.to_csv("building_summary.csv", index=False)
    df.to_csv("cleaned_energy_data.csv", index=False)

    print("Creating dashboard...")
    create_dashboard(df)

    print("Generating summary report...")
    generate_summary(df, summary_df)

    print("DONE! Files generated:")
    print("- dashboard.png")
    print("- cleaned_energy_data.csv")
    print("- building_summary.csv")
    print("- summary.txt")


if __name__ == "__main__":
    main()
