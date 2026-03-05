"""루트 conftest — --submission-dir CLI 옵션 등록"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--submission-dir",
        action="store",
        default=None,
        help="응시자 제출물 디렉토리 경로",
    )
