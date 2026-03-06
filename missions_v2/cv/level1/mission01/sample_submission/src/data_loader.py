"""data_loader.py - 다중 소스 데이터 로드 및 통합"""
import csv
import json
import os
import numpy as np
from datetime import datetime


def load_power_data(csv_paths):
    """복수의 CSV 파일을 로드하여 시간순으로 연결합니다."""
    all_rows = []
    for path in csv_paths:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt = row["datetime"]
                val = row["demand_kwh"]
                demand = float(val) if val.strip() != "" else np.nan
                all_rows.append([dt, demand])
    all_rows.sort(key=lambda x: x[0])
    return np.array(all_rows, dtype=object)


def load_weather_data(json_paths):
    """JSON 파일들을 로드하여 일별 기상 데이터를 통합합니다."""
    all_entries = []
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        all_entries.extend(data)
    all_entries.sort(key=lambda x: x["date"])
    result = []
    for e in all_entries:
        result.append([
            e["date"], e["avg_temp_c"], e["max_temp_c"],
            e["min_temp_c"], e["precipitation_mm"], e["wind_speed_ms"]
        ])
    return np.array(result, dtype=object)


def load_hourly_features(temp_tsv, humid_tsv):
    """TSV 파일에서 시간별 기온(F->C 변환)과 습도를 로드합니다."""
    temps = []
    with open(temp_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dt = row["datetime"]
            temp_f = float(row["temperature_f"])
            temp_c = (temp_f - 32) * 5 / 9
            temps.append([dt, round(temp_c, 2)])

    humids = {}
    with open(humid_tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            humids[row["datetime"]] = float(row["humidity_pct"])

    result = []
    for dt, temp_c in temps:
        humid = humids.get(dt, np.nan)
        result.append([dt, temp_c, humid])
    return np.array(result, dtype=object)


def load_holidays(csv_path):
    """공휴일 CSV를 로드합니다."""
    holidays = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            holidays.add(row["date"])
    return holidays


def merge_all(power, weather, hourly_features, holidays):
    """시간 기준으로 모든 데이터를 병합합니다."""
    # weather를 날짜별 dict로 변환
    weather_dict = {}
    for row in weather:
        weather_dict[str(row[0])] = [float(row[i]) for i in range(1, 6)]

    # hourly_features를 dict로 변환
    hourly_dict = {}
    for row in hourly_features:
        hourly_dict[str(row[0])] = [float(row[1]), float(row[2])]

    merged = []
    for row in power:
        dt_str = str(row[0])
        demand = row[1]
        date_str = dt_str[:10]

        # 기상 데이터
        w = weather_dict.get(date_str, [np.nan] * 5)
        # 시간별 기온/습도
        hf = hourly_dict.get(dt_str, [np.nan, np.nan])
        # 공휴일 플래그
        is_holiday = 1.0 if date_str in holidays else 0.0

        merged.append([dt_str, float(demand) if not isinstance(demand, float) or not np.isnan(demand) else np.nan]
                      + w + hf + [is_holiday])

    return np.array(merged, dtype=object)
