"""미션별 conftest — submission_dir, data_path fixture 제공"""
import json
import os
import zipfile

import pytest

_MISSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SUBMISSION = os.path.join(_MISSION_DIR, "sample_submission")

_DATA = {
    "images": {
        "img_01": [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ],
        "img_02": [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ],
        "img_03": [
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ]
    },
    "kernels": {
        "edge_h": [
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ],
        "edge_v": [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        "sharpen": [
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ]
    }
}


@pytest.fixture(scope="session")
def submission_dir(request, tmp_path_factory):
    """응시자 제출물 디렉토리 경로 (zip 파일 자동 해제 지원)"""
    cli_value = request.config.getoption("--submission-dir")
    resolved = os.path.abspath(cli_value) if cli_value else _DEFAULT_SUBMISSION

    if resolved.endswith(".zip"):
        assert os.path.isfile(resolved), f"zip 파일 없음: {resolved}"
        extract_dir = str(tmp_path_factory.mktemp("submission"))
        with zipfile.ZipFile(resolved, "r") as zf:
            zf.extractall(extract_dir)
        entries = os.listdir(extract_dir)
        if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
            extract_dir = os.path.join(extract_dir, entries[0])
        return extract_dir

    assert os.path.isdir(resolved), f"제출물 디렉토리 없음: {resolved}"
    return resolved


@pytest.fixture(scope="session")
def data_path(tmp_path_factory):
    """데이터를 임시 파일로 생성하여 경로 제공"""
    tmp_dir = tmp_path_factory.mktemp("data")
    path = str(tmp_dir / "data.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_DATA, f, ensure_ascii=False)
    return path
