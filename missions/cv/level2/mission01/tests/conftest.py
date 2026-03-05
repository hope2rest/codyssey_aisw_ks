"""미션별 conftest — submission_dir fixture 제공"""
import os

import pytest

_MISSION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_SUBMISSION = os.path.join(_MISSION_DIR, "sample_submission")


@pytest.fixture(scope="session")
def submission_dir(request):
    """응시자 제출물 디렉토리 경로"""
    cli_value = request.config.getoption("--submission-dir")
    resolved = os.path.abspath(cli_value) if cli_value else _DEFAULT_SUBMISSION
    assert os.path.isdir(resolved), f"제출물 디렉토리 없음: {resolved}"
    return resolved
