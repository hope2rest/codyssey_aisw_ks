import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_power_data, load_weather_data, load_hourly_features, load_holidays
from preprocessor import handle_missing, detect_outliers_iqr, validate_data
from feature_engineer import add_lag_features, add_rolling_features, add_time_features, add_holiday_flag
from model import split_time_series, train_linear, train_ridge, predict, evaluate, compare_models


def main(data_dir=None):
    # TODO: 전체 파이프라인 실행 및 result_q1.json 저장
    # 1. 데이터 로드 (10개 파일)
    # 2. 전처리 (결측 보간, 이상치 제거)
    # 3. 검증 (월별 통계 비교)
    # 4. 피처 엔지니어링 (lag, rolling, time, holiday)
    # 5. 모델 학습 (선형 회귀, Ridge 회귀)
    # 6. 평가 (MAE, RMSE, R², MAPE)
    # 7. result_q1.json 저장
    pass


if __name__ == "__main__":
    main()
