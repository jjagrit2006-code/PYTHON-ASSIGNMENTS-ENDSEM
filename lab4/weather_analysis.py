import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ------------ CONFIG ------------
DATA_PATH = "data/weather.csv" # your CSV inside data folder
print("DATA_PATH =", DATA_PATH, "type:", type(DATA_PATH))

# Match your CSV header exactly:
DATE_COL = "Date.Full"
TEMP_COL = "Data.Temperature.Avg Temp"
RAIN_COL = "Data.Precipitation"

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------ TASK 1: LOAD DATA ------------
df = pd.read_csv(DATA_PATH)

print("Head:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nDescribe:")
print(df.describe())

# ------------ TASK 2: CLEANING ------------
# Convert date column to datetime
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

# Drop rows with invalid dates
df = df.dropna(subset=[DATE_COL])

# Keep only relevant columns (no humidity in this dataset)
cols_needed = [DATE_COL, TEMP_COL, RAIN_COL]
df = df[cols_needed]

# Handle missing numeric values (fill with column mean)
for col in [TEMP_COL, RAIN_COL]:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].mean())

# Add helper columns
df = df.sort_values(DATE_COL)
df["Year"] = df[DATE_COL].dt.year
df["Month"] = df[DATE_COL].dt.month
df["Day"] = df[DATE_COL].dt.day

# ------------ TASK 3: STATISTICS (NumPy) ------------
temp = df[TEMP_COL].to_numpy()
rain = df[RAIN_COL].to_numpy()

stats = {
    "temp_mean": np.mean(temp), 
    "temp_min": np.min(temp),
    "temp_max": np.max(temp),
    "temp_std": np.std(temp),
    "rain_mean": np.mean(rain),
    "rain_min": np.min(rain),
    "rain_max": np.max(rain),
    "rain_std": np.std(rain),
}

print("\nOverall statistics:")
for k, v in stats.items():
    print(f"{k}: {v:.2f}")

# Daily stats
daily = df.set_index(DATE_COL).resample("D").agg({
    TEMP_COL: ["mean", "min", "max"],
    RAIN_COL: "sum"
})

# Monthly stats
monthly = df.set_index(DATE_COL).resample("M").agg({
    TEMP_COL: ["mean", "min", "max"],
    RAIN_COL: "sum"
})

# Yearly stats
yearly = df.set_index(DATE_COL).resample("Y").agg({
    TEMP_COL: ["mean", "min", "max"],
    RAIN_COL: "sum"
})

print("\nMonthly statistics:")
print(monthly.head())

# ------------ TASK 4: VISUALIZATION ------------
# Line chart for daily temperature trends
plt.figure(figsize=(10, 5))
plt.plot(daily.index, daily[(TEMP_COL, "mean")], label="Daily mean temperature")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Daily Temperature Trend")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "daily_temperature_line.png"))
plt.close()

# Bar chart for monthly rainfall totals
plt.figure(figsize=(8, 5))
plt.bar(monthly.index.strftime("%Y-%m"), monthly[(RAIN_COL, "sum")])
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Total Precipitation")
plt.title("Monthly Precipitation Totals")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "monthly_rainfall_bar.png"))
plt.close()

# Combined figure (two subplots: temp + rainfall)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
axes[0].plot(daily.index, daily[(TEMP_COL, "mean")], color="tab:red")
axes[0].set_ylabel("Temp")
axes[0].set_title("Temperature and Precipitation")

axes[1].bar(daily.index, daily[(RAIN_COL, "sum")], color="tab:blue")
axes[1].set_ylabel("Precipitation")
axes[1].set_xlabel("Date")

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "combined_temp_rain.png"))
plt.close()

# ------------ TASK 5: GROUPING & AGGREGATION ------------
grouped_month = df.groupby("Month").agg({
    TEMP_COL: ["mean", "min", "max"],
    RAIN_COL: "sum"
})

print("\nGrouped by month:")
print(grouped_month)

# Simple seasons by month
def month_to_season(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["Season"] = df["Month"].apply(month_to_season)
season_group = df.groupby("Season").agg({
    TEMP_COL: "mean",
    RAIN_COL: "sum"
})
print("\nGrouped by season:")
print(season_group)

# ------------ TASK 6: EXPORTS ------------
clean_csv_path = os.path.join(OUTPUT_DIR, "weather_cleaned.csv")
df.to_csv(clean_csv_path, index=False)
print(f"\nCleaned data saved to: {clean_csv_path}")

summary_lines = []
summary_lines.append("# Weather Data Summary\n")
summary_lines.append("This report briefly describes temperature and precipitation patterns.\n")
summary_lines.append("Overall mean temperature: {:.2f}\n".format(stats["temp_mean"]))
summary_lines.append("Overall mean precipitation: {:.2f}\n".format(stats["rain_mean"]))
summary_lines.append("\nMonthly overview (first few months):\n")
summary_lines.append(str(grouped_month.head()))

report_path = os.path.join(OUTPUT_DIR, "summary_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print(f"Summary report saved to: {report_path}")
print("All plots saved in the 'output' folder.")