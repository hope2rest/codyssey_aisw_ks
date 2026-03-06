## 문항 3 정답지 — 이미지 기반 객체 카운팅

### 정답 파일 구성

| 파일 | 주요 내용 |
|------|----------|
| `conv2d.py` | NumPy 기반 conv2d, Sobel 엣지 검출, 그레이스케일 변환, 반전, 밝기 조절, 정규화 |
| `counter.py` | 이미지 로드, 이진화, connected component, 앙상블 카운팅, 바운딩 박스 추출 |
| `metrics.py` | MAE/Accuracy 계산, worst case 탐색, 기본 vs 증강 비교 |
| `main.py` | 전체 파이프라인 실행, result_q3.json 저장 |

### 정답 체크리스트

| 번호 | 체크 항목 | 검증 방법 |
|------|----------|----------|
| 1 | conv2d.py 필수 함수 7개 | AST 분석 |
| 2 | filter2D 미사용 | 소스코드 검색 |
| 3 | counter.py 필수 함수 4개 + THRESHOLD/MIN_AREA | AST 분석 |
| 4 | metrics.py 필수 함수 3개 | AST 분석 |
| 5-9 | conv2d 기능 검증 | import + 수치 검증 |
| 10-12 | counter 기능 검증 | import + 수치 검증 |
| 13-15 | metrics 기능 검증 | import + 수치 검증 |
| 16-18 | result_q3.json 정량적 검증 | JSON 값 확인 |
