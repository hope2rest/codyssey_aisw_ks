# codyssey_aisw_0306

AI/SW 심화 시험 — pytest 기반 자동 채점 멀티 모듈 프로젝트

## 프로젝트 현황

| 항목 | 값 |
|------|-----|
| 총 미션 수 | 7개 (CS 2 + CV 5) |
| 총 테스트 수 | 135개 |
| 난이도 분포 | Level 1 × 4, Level 2 × 2, Level 3 × 1 |
| 채점 방식 | pytest 자동 채점 + `--submission-dir` 옵션 |

## 구현된 미션 목록

### CS (컴퓨터 사이언스)

| 문항 | 난이도 | 제목 | 테스트 수 | 배점 |
|------|--------|------|-----------|------|
| Q1 | Level 1 | MAC 연산 기반 패턴 매칭 | 11 | 100점 |
| Q2 | Level 2 | 2D 컨볼루션 기반 특징 추출 | 12 | 100점 |

### CV (컴퓨터 비전)

| 문항 | 난이도 | 제목 | 테스트 수 | 배점 |
|------|--------|------|-----------|------|
| Q1 | Level 1 | 이커머스 데이터 전처리 및 이상치 탐지 | 16 | 100점 |
| Q2 | Level 1 | 도서 검색 및 추천 서비스 | 30 | 100점 |
| Q3 | Level 2 | 이미지 객체 검출 및 데이터 증강을 통한 정확도 향상 | 14 | 100점 |
| Q4 | Level 2 | 금융 리스크 예측 모델 고도화 및 예측 시스템 | 18 | 100점 |
| Q5 | Level 3 | 금융 리스크 예측 서비스 | 34 | 100점 |

## 디렉토리 구조

```
codyssey_aisw_0306/
├── conftest.py                         # 루트: --submission-dir CLI 옵션 등록
├── pyproject.toml                      # pytest 설정
├── requirements.txt
├── README.md
└── missions_v2/
    ├── cs/
    │   ├── level1/mission01/           # CS Q1: MAC 연산 기반 패턴 매칭
    │   │   ├── problem.md
    │   │   ├── solution.md
    │   │   ├── data/
    │   │   │   └── data.json
    │   │   ├── template/
    │   │   │   └── mac_scorer.py
    │   │   ├── sample_submission/
    │   │   │   └── mac_scorer.py
    │   │   └── tests/
    │   │       ├── conftest.py
    │   │       └── test_mac_scorer.py
    │   └── level2/mission02/           # CS Q2: 2D 컨볼루션 기반 특징 추출
    │       ├── problem.md
    │       ├── solution.md
    │       ├── data/
    │       │   └── data.json
    │       ├── template/
    │       │   └── conv2d_analyzer.py
    │       ├── sample_submission/
    │       │   └── conv2d_analyzer.py
    │       └── tests/
    │           ├── conftest.py
    │           └── test_conv2d_analyzer.py
    └── cv/
        ├── level1/
        │   ├── mission01/              # CV Q1: 이커머스 데이터 전처리 + 이상치 탐지
        │   │   ├── problem.md
        │   │   ├── solution.md
        │   │   ├── data/
        │   │   │   └── customers.csv
        │   │   ├── template/
        │   │   │   └── q1_solution.py
        │   │   ├── sample_submission/
        │   │   │   ├── q1_solution.py
        │   │   │   └── result_q1.json
        │   │   └── tests/
        │   │       ├── conftest.py
        │   │       └── test_q1.py
        │   └── mission02/              # CV Q2: 도서 검색 및 추천 서비스
        │       ├── problem.md
        │       ├── solution.md
        │       ├── data/
        │       │   ├── books.txt, book_metadata.json
        │       │   ├── stopwords.txt, queries.txt
        │       │   ├── reviews.txt, sentiment_dict.json
        │       ├── template/
        │       │   ├── core/            # search_engine.py, recommender.py, sentiment.py, main.py
        │       │   ├── dashboard/       # app.py, pages/, components/, assets/
        │       │   ├── charts/          # search_charts.py, recommend_charts.py, sentiment_charts.py
        │       │   └── output/          # .gitkeep, charts/.gitkeep
        │       ├── sample_submission/
        │       │   ├── core/            # search_engine.py, recommender.py, sentiment.py, main.py
        │       │   ├── dashboard/       # app.py, pages/, components/, assets/
        │       │   ├── charts/          # search_charts.py, recommend_charts.py, sentiment_charts.py
        │       │   └── output/          # result_q2.json, charts/*.png
        │       └── tests/
        │           ├── conftest.py
        │           └── test_q2.py
        ├── level2/
        │   ├── mission03/              # CV Q3: 이미지 객체 검출 및 데이터 증강
        │   │   ├── problem.md
        │   │   ├── solution.md
        │   │   ├── data/
        │   │   │   ├── labels.json
        │   │   │   └── images/         # easy/medium/hard 각 5장 (15장)
        │   │   ├── template/
        │   │   │   ├── src/            # conv2d.py, counter.py, metrics.py, main.py
        │   │   │   ├── config/         # settings.json
        │   │   │   └── output/         # .gitkeep
        │   │   ├── sample_submission/
        │   │   │   ├── src/            # conv2d.py, counter.py, metrics.py, main.py
        │   │   │   ├── config/         # settings.json
        │   │   │   └── output/         # result_q3.json
        │   │   └── tests/
        │   │       ├── conftest.py
        │   │       └── test_q3.py
        │   └── mission04/              # CV Q4: 금융 리스크 예측 + 모델 해석
        │       ├── problem.md
        │       ├── solution.md
        │       ├── data/
        │       │   └── loan_data.csv
        │       ├── template/
        │       │   ├── src/            # preprocessor.py, model.py, interpreter.py, main.py
        │       │   ├── config/         # config.json
        │       │   ├── models/         # model_info.json
        │       │   └── output/         # .gitkeep
        │       ├── sample_submission/
        │       │   ├── src/            # preprocessor.py, model.py, interpreter.py, main.py
        │       │   ├── config/         # config.json
        │       │   ├── models/         # model_info.json
        │       │   └── output/         # result_q4.json
        │       └── tests/
        │           ├── conftest.py
        │           └── test_q4.py
        └── level3/
            └── mission05/              # CV Q5: 금융 리스크 예측 서비스
                ├── problem.md
                ├── solution.md
                ├── data/
                │   ├── loan_data.csv
                │   ├── new_customers.csv
                │   └── threshold_config.json
                ├── template/
                │   ├── core/           # preprocessor.py, model.py, interpreter.py, predictor.py, main.py
                │   ├── dashboard/      # app.py, pages/, components/, assets/
                │   ├── charts/         # risk_charts.py, feature_charts.py, pca_charts.py, cluster_charts.py
                │   ├── config/         # config.json
                │   ├── models/         # model_info.json
                │   └── output/         # .gitkeep, charts/.gitkeep
                ├── sample_submission/
                │   ├── core/           # preprocessor.py, model.py, interpreter.py, predictor.py, main.py
                │   ├── dashboard/      # app.py, pages/, components/, assets/
                │   ├── charts/         # risk_charts.py, feature_charts.py, pca_charts.py, cluster_charts.py
                │   ├── config/         # config.json
                │   ├── models/         # model_info.json
                │   └── output/         # result_q5.json, charts/*.png
                └── tests/
                    ├── conftest.py
                    └── test_q5.py
```

## 미션별 채점 명령어

```bash
# 정답 코드로 검증 (sample_submission 기본 사용)
pytest missions_v2/cs/level1/mission01/tests/ -v          # CS Q1: MAC 연산
pytest missions_v2/cs/level2/mission02/tests/ -v          # CS Q2: 2D 컨볼루션
pytest missions_v2/cv/level1/mission01/tests/ -v          # CV Q1: 이커머스 전처리
pytest missions_v2/cv/level1/mission02/tests/ -v          # CV Q2: TF-IDF 검색
pytest missions_v2/cv/level2/mission03/tests/ -v          # CV Q3: 객체 검출
pytest missions_v2/cv/level2/mission04/tests/ -v          # CV Q4: 리스크 예측
pytest missions_v2/cv/level3/mission05/tests/ -v          # CV Q5: 딥러닝 프레임워크

# 전체 미션 일괄 채점
pytest missions_v2/ -v

# 학생 제출물 채점 (--submission-dir 옵션)
pytest missions_v2/cv/level1/mission01/tests/ --submission-dir /path/to/submission -v
```

## 채점 메커니즘

### 디렉토리 기반 모듈 로딩

각 미션의 `tests/conftest.py`가 `--submission-dir` 옵션 또는 기본 `sample_submission/` 경로에서 학생 코드를 동적으로 import한다.

```
학생 제출 → conftest.py가 경로 해석 → sys.path 삽입 → 테스트 실행 → 점수 산출
```

### 테스트 패턴

| 검증 방법 | 설명 | 예시 |
|-----------|------|------|
| AST 자동 | 소스 코드 구조 검사 (필수 함수/클래스 존재 여부, 외부 라이브러리 사용 제한) | `functions_exist`, `no_external_lib` |
| import 자동 | 함수를 직접 호출하여 입출력 검증 | `test_mac_basic`, `test_relu` |
| JSON 자동 | 결과 JSON 파일의 키/값 구조 및 범위 검증 | `test_result_structure`, `test_sentiment_accuracy` |

### 배점 체계

- 각 미션별 100점 만점
- 개별 테스트 항목별 5~10점 배점 (세부 배점은 각 미션의 `solution.md` 참조)
- Pass 기준: 각 미션 100점 중 100점 (전체 정답)

## AI 트랩 설계 철학

각 미션에는 AI 코드 생성 도구가 흔히 범하는 실수를 유도하는 트랩이 포함되어 있다.

| 미션 | AI 트랩 |
|------|---------|
| CS Q1 | labels 키 대소문자 불규칙, 부동소수점 패턴, diagnose_failure 판별 순서 |
| CS Q2 | conv2d 자동 패딩 적용, relu in-place 수정, compute_stats 정수 나눗셈 |
| CV Q1 | Z-score 계산 시 표본 표준편차 vs 모집단 표준편차, IQR 경계값 포함/미포함 |
| CV Q2 | Smooth IDF 수식 오류 (log vs log10), 코사인 유사도 영벡터, 부정어 처리 순서 |
| CV Q3 | 컨볼루션 경계 처리, 바운딩 박스 병합 로직, 증강 이미지 정규화 누락 |
| CV Q4 | PCA 분산 비율 계산, K-Means 초기화 재현성, Feature Importance 정규화 |
| CV Q5 | 자동 미분 체인룰 누적, Bias-Variance 분해 공식, 학습률 스케줄링 경계값 |

## 새 미션 추가 방법

1. `missions_v2/{cs|cv}/level{N}/mission{NN}/` 디렉토리 생성
2. 필수 파일 배치:
   - `problem.md` — 문제 설명서
   - `solution.md` — 정답 코드 + 체크리스트 + 배점
   - `data/` — 입력 데이터
   - `template/` — 학생용 빈 템플릿
   - `sample_submission/` — 정답 코드 (채점 검증용)
   - `tests/conftest.py` — 제출물 로딩 설정
   - `tests/test_*.py` — pytest 테스트 케이스
3. `solution.md`에 정답 코드, 정답 체크리스트 (배점 포함), 데이터 타입, 학습 목표 매핑 포함
4. `pyproject.toml`의 `testpaths`에 경로가 포함되는지 확인

## 기술 스택

| 항목 | 버전/도구 |
|------|-----------|
| Python | 3.10+ |
| 테스트 프레임워크 | pytest >= 7.0 |
| 수치 연산 | NumPy (CV 미션) |
| 데이터 | JSON, CSV, TSV, NPZ |
| 채점 인터페이스 | CLI (`--submission-dir`) |
