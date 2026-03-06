def compute_metrics(predictions, labels, category):
    # TODO: 카테고리별 MAE, Accuracy 계산


def find_worst_case(predictions, labels, category):
    # TODO: 오차가 가장 큰 이미지 이름 반환


def get_failure_reasons():
    # TODO: 규칙 기반 실패 원인 3가지 이상 서술 (한국어, 20자+)


def get_why_learning_based():
    # TODO: 학습 기반 접근법 필요성 서술 (한국어, 30~200자)


def compare_methods(predictions_base, predictions_aug, labels):
    # TODO: 기본 vs 증강 방식의 카테고리별 MAE/Accuracy 비교


def create_detection_log(predictions, labels, date_str):
    # TODO: 일자별 검출 로그 생성 (date, total_images, results, daily_accuracy)


def generate_weekly_report(daily_logs):
    # TODO: 주간 요약 보고서 생성 (week_start/end, total_images_processed, average_daily_accuracy, best/worst_day)
