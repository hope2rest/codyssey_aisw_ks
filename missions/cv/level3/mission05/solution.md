## 문항 5 풀이 가이드: 자동차 부품 결함 검출 (딥러닝 기초 + 전이학습)

### 채점 기준 (10개 항목, 각 10점)

| 번호 | 체크 항목 | 배점 |
|------|-----------|------|
| 1 | 클래스 구현 (DefectImageLoader + InspectionLogProcessor) | 10 |
| 2 | 데이터 정제 결과 | 10 |
| 3 | conv2d 직접 구현 + Sobel 커널 | 10 |
| 4 | 규칙 기반 결과 | 10 |
| 5 | ML 기반 결과 | 10 |
| 6 | NN Forward Pass | 10 |
| 7 | 전이학습 비교 | 10 |
| 8 | 개선 실험 | 10 |
| 9 | 비즈니스 보고서 | 10 |
| 10 | 라이브러리 적절 사용 | 10 |

> **주의:** 10개 항목 모두 충족해야 Pass입니다.

### AI 트랩 주의사항

- **keras/torch 사용 금지**: sklearn만 사용 가능 (PCA, TfidfVectorizer, LogisticRegression)
- **class_weight='balanced'**: 개선 실험에서 반드시 적용
- **ReLU, Softmax 직접 구현**: NumPy로 직접 구현해야 하며, 수치 안정성 확보 필요 (Softmax에서 max 빼기)
- **pretrained_nn_weights.npz**: `W1`, `b1`, `W2`, `b2`, `feature_mean`, `feature_std` 가중치를 올바르게 로드

### Part A: 데이터 전처리 풀이

#### DefectImageLoader 구현

```python
from PIL import Image
import numpy as np

class DefectImageLoader:
    def __init__(self, image_dir, target_size=(64, 64)):
        self.image_dir = image_dir
        self.target_size = target_size

    def load_images(self):
        images = []
        valid_ids = []
        for i in range(500):
            fname = f"{i:04d}.png"
            fpath = os.path.join(self.image_dir, fname)
            try:
                img = Image.open(fpath)
                img.verify()  # 검증
                img = Image.open(fpath)  # verify 후 재오픈 필요
                img = img.convert("RGB")
                img = img.resize(self.target_size)
                arr = np.array(img, dtype=np.float64) / 255.0
                images.append(arr.flatten())  # 12288차원
                valid_ids.append(f"{i:04d}")
            except Exception:
                continue  # 손상 파일 건너뛰기
        return np.array(images), valid_ids
```

#### InspectionLogProcessor 구현

```python
import pandas as pd

class InspectionLogProcessor:
    VALID_LABELS = ["양품", "스크래치", "크랙", "변색", "이물질"]

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def process(self, valid_ids):
        df = self.df.copy()
        df = df.drop_duplicates(subset="part_id", keep="first")
        df["defect_type"] = df["defect_type"].str.strip()
        df = df[df["defect_type"].isin(self.VALID_LABELS)]
        df["inspector_note"] = df["inspector_note"].fillna("")
        df["part_id_str"] = df["part_id"].apply(lambda x: f"{x:04d}")
        df = df[df["part_id_str"].isin(valid_ids)]
        return df
```

#### 통계 출력

```python
label_dist = df["defect_type"].value_counts().to_dict()
imbalance_ratio = max(label_dist.values()) / min(label_dist.values())
```

### Part B: 3단계 모델 비교 풀이

#### B-1. 규칙 기반 (conv2d + Sobel)

```python
def conv2d(image, kernel):
    """NumPy 직접 구현 - valid 모드"""
    ih, iw = image.shape
    kh, kw = kernel.shape
    oh, ow = ih - kh + 1, iw - kw + 1
    output = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return output

# Sobel 커널
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)

def compute_edge_magnitude(image_flat):
    img = image_flat.reshape(64, 64, 3)
    gray = np.mean(img, axis=2)  # Grayscale 변환
    gx = conv2d(gray, sobel_x)
    gy = conv2d(gray, sobel_y)
    magnitude = np.sqrt(gx**2 + gy**2)
    return np.mean(magnitude)
```

- threshold: train 데이터 양품 클래스의 엣지 강도 **중앙값**
- 이진 분류: 엣지 강도 <= threshold → 양품(0), 그 외 → 불량(1)

#### B-2. ML 기반 (PCA + TF-IDF + LogisticRegression)

```python
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# PCA (95% 분산 설명)
pca = PCA(n_components=0.95, random_state=42)
X_pca_train = pca.fit_transform(X_train_img)
X_pca_test = pca.transform(X_test_img)

# TF-IDF
tfidf = TfidfVectorizer(max_features=100)
X_tfidf_train = tfidf.fit_transform(train_notes).toarray()
X_tfidf_test = tfidf.transform(test_notes).toarray()

# 결합
X_train_combined = np.hstack([X_pca_train, X_tfidf_train])
X_test_combined = np.hstack([X_pca_test, X_tfidf_test])

# 학습
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_combined, y_train)
```

#### B-3. 2층 NN Forward Pass (NumPy 직접 구현)

```python
# 가중치 로드
weights = np.load("pretrained_nn_weights.npz")
W1, b1 = weights["W1"], weights["b1"]
W2, b2 = weights["W2"], weights["b2"]
feat_mean = weights["feature_mean"]
feat_std = weights["feature_std"]

# 특징 로드 및 정규화
features = np.load("pretrained_features.npy")
X_norm = (features - feat_mean) / (feat_std + 1e-8)

# ReLU 직접 구현
def relu(x):
    return np.maximum(0, x)

# Softmax 직접 구현 (수치 안정성)
def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Forward Pass
z1 = X_norm @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
probs = softmax(z2)
predictions = np.argmax(probs, axis=1)
```

> **주의:** Softmax 구현 시 `x - np.max(x, ...)` 처리로 수치 오버플로 방지가 필수입니다.

### Part C: 전이학습 비교 풀이

```python
# Scratch: PCA 이미지 특징만 (텍스트 제외)
lr_scratch = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_scratch.fit(X_pca_train, y_train)
scratch_acc = lr_scratch.score(X_pca_test, y_test)

# Pretrained: pretrained_features.npy 사용
lr_pretrained = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr_pretrained.fit(pretrained_train, y_train)
pretrained_acc = lr_pretrained.score(pretrained_test, y_test)

transfer_gain = round(pretrained_acc - scratch_acc, 4)
```

> **주의:** `pretrained_features.npy`의 인덱스는 원본 part_id 순서(0000~0499)에 대응하므로, train/test 분할 시 인덱스 매핑에 유의합니다.

### Part D: 성능 평가 + 개선 + 보고서 풀이

#### 성능 평가

```python
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Macro F1
f1_macro = round(f1_score(y_test, y_pred, average="macro"), 4)

# 클래스별 F1
class_f1 = f1_score(y_test, y_pred, average=None)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred).tolist()
```

#### 개선 실험

```python
# class_weight='balanced' 적용
lr_improved = LogisticRegression(
    C=1.0, max_iter=1000, random_state=42,
    class_weight="balanced"
)
lr_improved.fit(pretrained_train, y_train)
```

- 개선 전/후 Macro F1 비교
- 클래스별 F1 차이를 계산하여 가장 많이 개선된 클래스 보고

#### 비즈니스 보고서

4개 섹션 각 3문장 이내로 작성:
- `purpose`: 자동차 부품 품질관리 자동화 목적 및 불량 조기 검출 효과
- `key_results`: 각 모델의 정확도/F1 수치 요약
- `transfer_learning_effect`: 사전학습 특징 활용 시 성능 향상 정도와 의미
- `improvement_suggestion`: 데이터 증강, 더 깊은 모델, 실시간 추론 등 제안

### 최종 출력 구조

모든 수치는 소수점 이하 4자리로 반올림하여 `result_q5.json`에 저장합니다. `round(value, 4)`를 일관적으로 적용하세요.
