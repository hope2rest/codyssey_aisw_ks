THRESHOLD = 30
MIN_AREA = 100


def count_boxes(image_path, threshold=THRESHOLD, min_area=MIN_AREA):
    # TODO: 이미지 로드 → 엣지 검출 → 이진화 → Connected Component → 최소 면적 필터


def ensemble_count(counts):
    # TODO: 여러 카운팅 결과의 중앙값 반환 (정수)


def count_boxes_augmented(image_path, threshold=THRESHOLD, min_area=MIN_AREA):
    # TODO: 원본 + 증강 이미지에 대해 앙상블 카운팅


def extract_bounding_boxes(image_path, threshold=THRESHOLD, min_area=MIN_AREA):
    # TODO: 검출된 박스의 바운딩 박스 좌표(x_min, y_min, x_max, y_max, area) 추출
