"""data_loader.py - 다중 소스 데이터 로드 및 통합"""
import csv
import json
import os
import numpy as np
from datetime import datetime


def load_power_data(csv_paths):
    """복수의 CSV 파일을 로드하여 시간순으로 연결합니다."""
    # TODO: CSV 파일들 로드, datetime 파싱, demand_kwh float 변환
    pass


def load_weather_data(json_paths):
    """JSON 파일들을 로드하여 일별 기상 데이터를 통합합니다."""
    # TODO: JSON 파일들 로드, 날짜순 정렬
    pass


def load_hourly_features(temp_tsv, humid_tsv):
    """TSV 파일에서 시간별 기온(F->C 변환)과 습도를 로드합니다."""
    # TODO: TSV 로드, 화씨→섭씨 변환 (°C = (°F - 32) × 5/9)
    pass


def load_holidays(csv_path):
    """공휴일 CSV를 로드합니다."""
    # TODO: holidays.csv 로드, 날짜 set 반환
    pass


def merge_all(power, weather, hourly_features, holidays):
    """시간 기준으로 모든 데이터를 병합합니다."""
    # TODO: 시간별 전력 + 일별 기상 + 시간별 기온/습도 + 공휴일 플래그 병합
    pass
