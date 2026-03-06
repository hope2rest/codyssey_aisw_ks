# codyssey_aisw_0306

AI/SW 심화 시험 — pytest 기반 자동 채점 멀티 모듈 프로젝트

## 문제 개요

### CS (컴퓨터 사이언스)

| 문항 | 난이도 | 제목 | 설명 |
|------|--------|------|------|
| Q1 | Level 1 | MAC 연산 기반 패턴 매칭 | 3×3 패턴과 필터를 읽어 MAC 연산으로 최적 매칭을 찾고, 라벨 정규화·epsilon 비교·시간 복잡도 분석·실패 진단까지 수행 |
| Q2 | Level 2 | 2D 컨볼루션 기반 특징 추출 | 5×5 이미지와 3×3 커널로 2D 컨볼루션 특징맵을 추출하고, ReLU 활성화·통계 분석·최강 특징 커널 탐색 수행 |

### CV (컴퓨터 비전)

| 문항 | 난이도 | 제목 | 설명 |
|------|--------|------|------|
| Q1 | Level 1 | 이커머스 데이터 전처리 및 이상치 탐지 | 고객 데이터를 전처리하고, IQR/Z-score 기반 이상치 탐지 및 고객 세그먼트 분류 |
| Q2 | Level 1 | TF-IDF 문서 검색 + 규칙 기반 감성 분석 | TF-IDF 기반 문서 검색과 부정어/강조어 처리 포함 규칙 기반 감성 분석 |
| Q3 | Level 2 | 이미지 기반 객체 카운팅 | 2D 컨볼루션 엣지 검출, 데이터 증강 앙상블, 바운딩 박스 추출 및 성능 비교 |
| Q4 | Level 2 | 금융 리스크 예측 + 모델 해석 | ML 기반 리스크 예측 모델 구축, PCA/K-Means/Feature Importance 해석 |
| Q5 | Level 3 | 미니 딥러닝 프레임워크 + 성능 진단 | NumPy 기반 Tensor 자동 미분 엔진, 신경망 레이어, Bias/Variance 진단 |

## 디렉토리 구조

```
codyssey_aisw_0306/
├── conftest.py                         # 루트: --submission-dir CLI 옵션 등록
├── pyproject.toml                      # pytest 설정
├── requirements.txt
├── README.md
└── missions/
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
        │   │   ├── config.yaml
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
        │   └── mission02/              # CV Q2: TF-IDF 문서 검색 + 감성 분석
        │       ├── problem.md
        │       ├── solution.md
        │       ├── config.yaml
        │       ├── data/
        │       │   ├── documents.txt
        │       │   ├── queries.txt
        │       │   ├── reviews.txt
        │       │   ├── sentiment_dict.json
        │       │   └── stopwords.txt
        │       ├── template/
        │       │   └── q2_solution.py
        │       ├── sample_submission/
        │       │   ├── q2_solution.py
        │       │   └── result_q2.json
        │       └── tests/
        │           ├── conftest.py
        │           └── test_q2.py
        ├── level2/
        │   ├── mission03/              # CV Q3: 이미지 기반 객체 카운팅
        │   │   ├── problem.md
        │   │   ├── solution.md
        │   │   ├── data/
        │   │   │   ├── labels.json
        │   │   │   └── images/         # easy/medium/hard 각 5장 (15장)
        │   │   ├── template/
        │   │   │   ├── conv2d.py
        │   │   │   ├── counter.py
        │   │   │   ├── metrics.py
        │   │   │   └── main.py
        │   │   ├── sample_submission/
        │   │   │   ├── conv2d.py
        │   │   │   ├── counter.py
        │   │   │   ├── metrics.py
        │   │   │   ├── main.py
        │   │   │   └── result_q3.json
        │   │   └── tests/
        │   │       ├── conftest.py
        │   │       └── test_q3.py
        │   └── mission04/              # CV Q4: 금융 리스크 예측 + 모델 해석
        │       ├── problem.md
        │       ├── solution.md
        │       ├── config.yaml
        │       ├── data/
        │       │   └── loan_data.csv
        │       ├── template/
        │       │   ├── preprocessor.py
        │       │   ├── model.py
        │       │   ├── interpreter.py
        │       │   └── main.py
        │       ├── sample_submission/
        │       │   ├── preprocessor.py
        │       │   ├── model.py
        │       │   ├── interpreter.py
        │       │   ├── main.py
        │       │   └── result_q4.json
        │       └── tests/
        │           ├── conftest.py
        │           └── test_q4.py
        └── level3/
            └── mission05/              # CV Q5: 미니 딥러닝 프레임워크 + 성능 진단
                ├── problem.md
                ├── solution.md
                ├── config.yaml
                ├── data/
                │   ├── xor_data.npz
                │   ├── regression_data.npz
                │   └── generate_data.py
                ├── template/
                │   ├── tensor.py
                │   ├── autograd.py
                │   ├── layers.py
                │   ├── trainer.py
                │   ├── diagnostics.py
                │   └── main.py
                ├── sample_submission/
                │   ├── tensor.py
                │   ├── autograd.py
                │   ├── layers.py
                │   ├── trainer.py
                │   ├── diagnostics.py
                │   ├── main.py
                │   └── result_q5.json
                └── tests/
                    ├── conftest.py
                    └── test_q5.py
```

## 실행 방법

```bash
# 정답 코드로 검증 (sample_submission 기본 사용)
pytest missions/cs/level1/mission01/tests/ -v
pytest missions/cs/level2/mission02/tests/ -v
pytest missions/cv/level1/mission01/tests/ -v
pytest missions/cv/level1/mission02/tests/ -v
pytest missions/cv/level2/mission03/tests/ -v
pytest missions/cv/level2/mission04/tests/ -v
pytest missions/cv/level3/mission05/tests/ -v

# 학생 제출물 채점
pytest missions/cv/level1/mission01/tests/ --submission-dir /path/to/submission -v
```
