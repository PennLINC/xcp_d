import pytest


def pytest_addoption(parser):
    parser.addoption("--working_dir", action="store", default="/tmp")
    parser.addoption("--data_dir", action="store")
    parser.addoption("--output_dir", action="store")


# Set up the commandline options as fixtures
@pytest.fixture
def data_dir(request):
    return request.config.getoption("--data_dir")


@pytest.fixture
def working_dir(request):
    return request.config.getoption("--working_dir")


@pytest.fixture
def output_dir(request):
    return request.config.getoption("--output_dir")
