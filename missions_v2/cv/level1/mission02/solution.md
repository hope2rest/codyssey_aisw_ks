## 문항 2 정답지 — TF-IDF 문서 검색 + 규칙 기반 감성 분석

### 정답 체크리스트

| 번호 | 체크 항목 | 검증 방법 |
|------|----------|----------|
| 1 | 필수 함수 6개 정의 | AST 분석 |
| 2 | sklearn/scipy 미사용 | 소스 코드 검색 |
| 3 | NFC 정규화 적용 | 분해형/조합형 한글 동일 처리 확인 |
| 4 | 코사인 유사도 | 직교/동일/영벡터 케이스 검증 |
| 5 | 감성 예측 정확도 | 긍정/부정/부정어 처리 검증 |
| 6 | 메트릭 계산 | Accuracy/Precision/Recall/F1 수치 검증 |
| 7 | result 구조 | 10개 필수 키 확인 |
| 8 | result 값 | vocab_size=290, total_reviews=30 등 |

- **AI 트랩**:
  - NFC 정규화 순서: normalize -> lowercase -> 특수문자 제거
  - Smooth IDF: `log((N+1)/(df+1)) + 1` (일반 IDF가 아닌)
  - 부정어 처리: 바로 다음 토큰의 점수에 -1 곱함
  - positive 사전에서 먼저 조회 (positive/negative 중복 단어)
