"""
MarineMine Pro — Sample Dataset Generator
=========================================
Generates a realistic synthetic marine pollution dataset with 1,000 records,
correlated environmental parameters, realistic geographic coordinates,
realistic missing values (~8%), and controlled duplicates (~3%).
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_sample_dataset(n_records: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    regions_data = [
        {"region": "North Atlantic", "country": "United States", "lat_range": (25.0, 55.0), "lon_range": (-75.0, -20.0), "depth_range": (50, 4200)},
        {"region": "South Pacific", "country": "Australia", "lat_range": (-40.0, -10.0), "lon_range": (140.0, 175.0), "depth_range": (30, 5000)},
        {"region": "Indian Ocean", "country": "India", "lat_range": (-10.0, 22.0), "lon_range": (60.0, 95.0), "depth_range": (20, 3800)},
        {"region": "Arctic Sea", "country": "Norway", "lat_range": (68.0, 82.0), "lon_range": (10.0, 45.0), "depth_range": (40, 2500)},
        {"region": "Mediterranean Sea", "country": "Italy", "lat_range": (32.0, 44.0), "lon_range": (0.0, 32.0), "depth_range": (15, 2900)},
        {"region": "Caribbean Sea", "country": "Jamaica", "lat_range": (10.0, 22.0), "lon_range": (-85.0, -62.0), "depth_range": (10, 4500)},
        {"region": "South China Sea", "country": "Vietnam", "lat_range": (5.0, 22.0), "lon_range": (105.0, 119.0), "depth_range": (20, 4000)},
        {"region": "North Sea", "country": "United Kingdom", "lat_range": (52.0, 61.0), "lon_range": (-2.0, 8.0), "depth_range": (15, 600)},
        {"region": "Red Sea", "country": "Saudi Arabia", "lat_range": (15.0, 28.0), "lon_range": (35.0, 42.0), "depth_range": (20, 2100)},
        {"region": "Gulf of Mexico", "country": "Mexico", "lat_range": (20.0, 29.0), "lon_range": (-96.0, -82.0), "depth_range": (10, 3700)},
    ]

    pollution_types = [
        "Microplastics", "Oil Spill", "Heavy Metals", "Nutrient Pollution",
        "Chemical Discharge", "Sewage", "Thermal Pollution", "Noise Pollution",
        "Radioactive Waste", "Agricultural Runoff"
    ]

    units_map = {
        "Microplastics": "particles/m3",
        "Oil Spill": "ppm",
        "Heavy Metals": "mg/L",
        "Nutrient Pollution": "mg/L",
        "Chemical Discharge": "ug/L",
        "Sewage": "CFU/100mL",
        "Thermal Pollution": "delta_degC",
        "Noise Pollution": "dB",
        "Radioactive Waste": "Bq/L",
        "Agricultural Runoff": "mg/L",
    }

    weather_conditions = ["Clear", "Partly Cloudy", "Overcast", "Rainy", "Stormy", "Foggy"]
    ocean_conditions = ["Calm", "Moderate Swell", "Rough", "Very Rough", "Heavy Swell"]
    data_sources = ["Autonomous Buoy", "Research Vessel", "Satellite SAR", "Coastal Station", "Drifter Glider"]

    rows = []
    base_date = pd.Timestamp("2023-01-01")

    for i in range(n_records):
        reg_info = np.random.choice(regions_data)
        p_type = np.random.choice(pollution_types)

        lat = np.random.uniform(reg_info["lat_range"][0], reg_info["lat_range"][1])
        lon = np.random.uniform(reg_info["lon_range"][0], reg_info["lon_range"][1])
        depth = np.random.uniform(reg_info["depth_range"][0], reg_info["depth_range"][1])

        # Date over 2 years
        day_offset = np.random.randint(0, 730)
        obs_date = base_date + pd.Timedelta(days=day_offset)

        # Environmental correlations
        if reg_info["region"] == "Arctic Sea":
            temp = np.random.uniform(-1.8, 6.0)
            salinity = np.random.uniform(30.0, 34.0)
        elif reg_info["region"] in ["Red Sea", "Mediterranean Sea", "Caribbean Sea"]:
            temp = np.random.uniform(20.0, 31.0)
            salinity = np.random.uniform(36.0, 40.5)
        else:
            temp = np.random.uniform(8.0, 26.0)
            salinity = np.random.uniform(33.0, 36.5)

        ph = np.random.normal(8.1, 0.25)
        # Dissolved oxygen is inversely related to temperature
        dissolved_oxygen = max(2.0, 14.0 - 0.25 * temp + np.random.normal(0, 0.6))

        # Pollution value with skewed distribution & occasional high hotspots
        if p_type == "Microplastics":
            val = np.random.exponential(120.0) + 10.0
        elif p_type == "Oil Spill":
            val = np.random.exponential(80.0) + (400.0 if np.random.rand() < 0.08 else 5.0)
        elif p_type == "Heavy Metals":
            val = np.random.gamma(shape=2.0, scale=8.0) + 0.5
        elif p_type == "Nutrient Pollution":
            val = np.random.gamma(shape=3.0, scale=12.0) + 2.0
        elif p_type == "Sewage":
            val = np.random.exponential(150.0) + 15.0
        else:
            val = np.random.exponential(60.0) + 3.0

        val = round(min(999.0, val), 2)
        unit = units_map[p_type]

        rows.append({
            "record_id": f"MM-{i+1:05d}",
            "observation_date": obs_date.strftime("%Y-%m-%d"),
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "region": reg_info["region"],
            "country": reg_info["country"],
            "pollution_type": p_type,
            "pollution_value": val,
            "pollution_unit": unit,
            "temperature_celsius": round(temp, 2),
            "salinity_ppt": round(salinity, 2),
            "ph_level": round(ph, 2),
            "dissolved_oxygen": round(dissolved_oxygen, 2),
            "weather_condition": np.random.choice(weather_conditions),
            "ocean_condition": np.random.choice(ocean_conditions),
            "water_depth_m": round(depth, 1),
            "monitoring_station": f"{reg_info['region'][:3].upper()}-STN{np.random.randint(101, 150)}",
            "data_source": np.random.choice(data_sources),
            "notes": "SYNTHETIC DEMONSTRATION DATA",
        })

    df = pd.DataFrame(rows)

    # Introduce ~3% duplicate rows
    n_dups = int(n_records * 0.03)
    dup_indices = np.random.choice(len(df), size=n_dups, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Introduce ~8% missing values in non-essential columns
    for col in ["temperature_celsius", "salinity_ppt", "ph_level", "dissolved_oxygen", "weather_condition", "water_depth_m"]:
        mask = np.random.rand(len(df)) < 0.08
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "marine_pollution_sample.csv"
    df = generate_sample_dataset(1000)
    df.to_csv(out_path, index=False)
    print(f"Sample dataset generated: {out_path} ({len(df)} rows)")
