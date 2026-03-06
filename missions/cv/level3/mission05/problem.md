## 문항 5: 자동차 부품 결함 검출 (딥러닝 기초 + 전이학습)

### 문제

부품 이미지(`part_images/`)와 검사 기록(`inspection_log.csv`)을 활용하여 부품의 결함 유형을 3가지 방식(규칙 기반/ML/NN)으로 자동 분류하고 비교하는 시스템을 구축합니다.
- **데이터 위치**: `data/part_images/` (0000~0499.png), `data/inspection_log.csv`, `data/pretrained_features.npy`, `data/pretrained_nn_weights.npz`
- **핵심 개념**: 규칙 기반 엣지 검출, PCA+TF-IDF 특징 결합, 2층 신경망 Forward Pass(NumPy 직접 구현), 전이학습 효과 비교

> **주의:**
> - 레이블 분포는 **불균형** 상태입니다 (양품이 다수).
> - 이미지 일부에 **손상된 파일**이 포함되어 있을 수 있습니다.

### 입력 데이터

| 파일/폴더 | 설명 |
|-----------|------|
| `part_images/` | `0000.png` ~ `0499.png` (64x64, 일부 손상 포함) |
| `inspection_log.csv` | `part_id`, `defect_type`, `inspector_note` |
| `pretrained_features.npy` | 사전학습 특징 벡터 (500x128) |
| `pretrained_nn_weights.npz` | 2층 NN 가중치 (`W1`, `b1`, `W2`, `b2`, `feature_mean`, `feature_std`) |

### 구현 요구사항

#### Part A: 데이터 전처리 (클래스 기반 구현)

1. **`DefectImageLoader` 클래스**를 구현하세요:
   - 이미지를 **RGB 모드**로 변환 (Grayscale/RGBA 등 대응)
   - **64x64**로 리사이즈
   - **[0, 1] 범위**로 정규화 (`/255.0`)
   - 손상된 이미지(열 수 없거나 검증 실패)는 **건너뛰기**
   - 유효한 이미지만 **flatten**하여 반환 (64x64x3 = 12288차원)

2. **`InspectionLogProcessor` 클래스**를 구현하세요:
   - `inspection_log.csv` 로드
   - **중복 `part_id`** 제거 (`keep='first'`)
   - `defect_type` 전처리 (유효하지 않은 값 정리)
   - 유효 레이블: `["양품", "스크래치", "크랙", "변색", "이물질"]`에 해당하지 않는 행 제거
   - `inspector_note`의 **결측값(NaN)**을 빈 문자열(`""`)로 대체
   - 이미지가 유효한 `part_id`만 남기기

3. **출력**: 정제 후 유효 샘플 수, 레이블별 분포, 불균형 비율(최대/최소)

#### Part B: 3단계 모델 비교

4. **데이터 분할**: train(70%) / test(30%), `random_state=42`, `stratify=labels`

5. **B-1. 규칙 기반 (엣지 강도)**:
   - **NumPy**로 `conv2d(image, kernel)` 함수 직접 구현 (valid 모드)
   - Sobel 커널(3x3)로 수평/수직 엣지 검출 → `edge_magnitude = sqrt(Gx^2 + Gy^2)`
   - 각 이미지의 **엣지 평균 강도**를 계산
   - 임계값 기반 분류: 엣지 강도 <= `threshold` → 양품(0), 그 외 → 불량(1) (**이진 분류**)
   - `threshold`는 train 데이터 양품 클래스의 엣지 강도 **중앙값**으로 설정

6. **B-2. ML 기반 (특징 추출 + LR)**:
   - **이미지 특징**: flatten → PCA (`n_components` = 95% 분산 설명)
   - **텍스트 특징**: `inspector_note`에 대해 TF-IDF (`TfidfVectorizer`, `max_features=100`)
   - 이미지 + 텍스트 특징을 **수평 결합** (`np.hstack`)
   - `LogisticRegression(C=1.0, max_iter=1000, random_state=42)` 학습 (**5클래스 분류**)

7. **B-3. 2층 신경망 Forward Pass (NumPy 직접 구현)**:
   - `pretrained_nn_weights.npz`에서 가중치 로드
   - `pretrained_features.npy`에서 특징 벡터 로드
   - 2층 신경망의 Forward Pass를 NumPy로 직접 구현
   - 적절한 활성화 함수와 출력층 함수를 사용
   - 수치 안정성에 유의

#### Part C: 전이학습 비교

8. **Scratch vs Pretrained 비교**:
   - **Scratch 모델**: Part B-2의 PCA 이미지 특징(**텍스트 제외**) → `LogisticRegression` 학습
   - **Pretrained 모델**: `pretrained_features.npy`의 특징 → 같은 설정의 `LogisticRegression` 학습
   - 두 모델의 **test Accuracy 차이**를 `transfer_gain`으로 산출

> **주의:** `pretrained_features.npy`의 인덱스는 원본 `part_id` 순서(0000~0499)에 대응합니다.

#### Part D: 성능 평가 + 개선 + 비즈니스 보고서

9. **성능 평가**:
   - 모든 모델의 **test Accuracy**
   - ML(B-2), Pretrained(C) 모델의 **Macro F1-score**
   - Pretrained 모델의 **클래스별 F1-score** (5개)
   - Pretrained 모델의 **Confusion Matrix** (5x5, 리스트의 리스트)

10. **개선 실험**:
    - 소수 클래스 분류 성능을 개선하는 기법을 적용하여, 적용 전/후 **Macro F1**을 비교
    - 어떤 클래스의 F1이 가장 많이 **개선**되었는지 보고

11. **비즈니스 보고서** (4개 섹션, 각 3문장 이내):
    - `purpose`: 이 시스템의 목적과 기대 효과
    - `key_results`: 핵심 수치 결과 (정확도, F1 등)
    - `transfer_learning_effect`: 전이학습의 효과와 의미
    - `improvement_suggestion`: 향후 개선 방향 제안

### 출력 형식

`result_q5.json` 파일로 다음 구조를 저장하세요:

```json
{
  "data_summary": {
    "total_valid_samples": 정수,
    "label_distribution": {"양품": 정수, "스크래치": 정수, "크랙": 정수, "변색": 정수, "이물질": 정수},
    "imbalance_ratio": 실수
  },
  "rule_based": {
    "test_accuracy": 실수,
    "method": "edge_threshold_binary"
  },
  "ml_based": {
    "test_accuracy": 실수,
    "test_f1_macro": 실수,
    "pca_n_components": 정수
  },
  "nn_forward": {
    "test_accuracy": 실수,
    "test_f1_macro": 실수
  },
  "pretrained": {
    "test_accuracy": 실수,
    "test_f1_macro": 실수,
    "class_f1": {"양품": 실수, "스크래치": 실수, "크랙": 실수, "변색": 실수, "이물질": 실수},
    "confusion_matrix": [[정수 5개], ...]
  },
  "transfer_gain": 실수,
  "improvement": {
    "before_f1": 실수,
    "after_f1": 실수,
    "most_improved_class": "문자열"
  },
  "report": {
    "purpose": "3문장 이내",
    "key_results": "3문장 이내",
    "transfer_learning_effect": "3문장 이내",
    "improvement_suggestion": "3문장 이내"
  }
}
```

### 제약 사항

- `conv2d` 함수는 **NumPy만으로 직접 구현** (`cv2.filter2D` 등 사용 금지)
- 2층 신경망 Forward Pass는 **NumPy만으로 직접 구현** (PyTorch/Keras 금지)
- 이미지 로드에는 `PIL` 사용 가능
- `PCA`, `TfidfVectorizer`, `LogisticRegression`은 `sklearn` 사용 허용
- 모든 수치는 **소수점 이하 4자리**로 반올림

### 제출 방식

- `q5_solution.py`와 `result_q5.json` 두 파일을 제출
- 두 파일 모두 제출해야 채점이 진행됩니다
