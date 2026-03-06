"""미션별 conftest — submission_dir fixture 제공"""
import os
import tempfile
import zipfile

import pytest

_MISSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SUBMISSION = os.path.join(_MISSION_DIR, "sample_submission")



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
