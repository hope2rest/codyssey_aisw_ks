# Q5 정답: 딥러닝 프레임워크 설계를 통한 성능 검증

## 모듈 구조

| 파일 | 역할 |
|------|------|
| tensor.py | 역전파(reverse-mode autodiff) 지원 Tensor 클래스 |
| autograd.py | 수치적 그래디언트 검증 (central difference) |
| layers.py | Linear, Sequential, ReLU, Sigmoid, MSE, BCE |
| trainer.py | SGD 옵티마이저, 학습 루프 |
| diagnostics.py | Bias/Variance 진단, 학습 곡선 |
| main.py | 전체 파이프라인, result_q5.json 출력 |

## 핵심 구현 사항

### tensor.py - 역전파
- DFS 기반 위상 정렬 후 역순 순회
- 그래디언트 누적은 `+=` 사용 (`=` 사용 시 공유 노드 오류)
- add/mul backward에서 브로드캐스트된 축에 대해 합산 처리
- `sigmoid`: 수치 안정성을 위해 입력을 [-500, 500]으로 클리핑
- `log`: 입력을 [1e-12, inf]로 클리핑

### layers.py - 초기화 전략
- `'zero'`: 모두 0 (대칭이 깨지지 않아 학습 실패)
- `'random'`: N(0, 0.01) (기울기 소실 가능)
- `'he'`: N(0, sqrt(2/fan_in)) (ReLU 네트워크에 권장)

### diagnostics.py - Bias/Variance 진단
- `train_loss > threshold` → high_bias (과소적합)
- `test_loss - train_loss > threshold` → high_variance (과적합)
- 그 외 → good_fit

## 기대 결과

- XOR 정확도: 1.0 (4개 패턴 모두 정답)
- Gradient check: 최대 상대 오차 ≈ 0 (통과)
- 회귀 R-squared: >0.99
- 초기화 비교: he >> random >> zero
- 회귀 진단: good_fit

## 채점 기준 (정량적 평가)

| 검증 항목 | 배점 | 기준 |
|----------|------|------|
| 구조 검증 (AST) | 15 | 3개 테스트 통과 |
| Tensor 연산 | 25 | 4개 테스트 통과 (값 + 그래디언트) |
| 레이어 | 15 | 2개 테스트 통과 (shape + 체이닝) |
| 학습 루프 | 10 | 1개 테스트 통과 (손실 감소) |
| 성능 진단 | 15 | 2개 테스트 통과 (진단 문자열 + grad check) |
| 결과 JSON | 20 | 2개 테스트 통과 (구조 + xor_accuracy >= 0.75) |
