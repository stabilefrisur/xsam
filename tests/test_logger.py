from pathlib import Path

import pytest

from xsam.logger import ActionLogger, FileLogger, set_log_path


# Helper to fully reset logger singletons for test isolation
def reset_logger_singletons():
    import xsam.logger as xsam_logger

    xsam_logger._action_logger = None
    xsam_logger._file_logger = None
    # Remove initialized attribute if present
    if hasattr(ActionLogger, "_instance"):
        ActionLogger._instance = None
    if hasattr(FileLogger, "_instance"):
        FileLogger._instance = None


@pytest.fixture
def action_logger(tmp_path):
    log_file = tmp_path / "action_log"
    logger = ActionLogger(log_file)
    yield logger
    logger.logger.handlers.clear()
    logger.initialized = False


@pytest.fixture
def file_logger(tmp_path):
    log_file = tmp_path / "file_log"
    logger = FileLogger(log_file)
    yield logger
    logger.logger.handlers.clear()
    logger.initialized = False


def test_action_logger_info_and_warning(action_logger):
    action_logger.info("Test info message")
    action_logger.warning("Test warning message")
    log_path = action_logger.log_file
    assert log_path.exists()
    with log_path.open("r") as f:
        logs = f.read()
    assert "Test info message" in logs
    assert "Test warning message" in logs


def test_action_logger_stdout_and_file(tmp_path):
    reset_logger_singletons()
    log_file = tmp_path / "stdout_file_log"
    logger = ActionLogger(log_file)
    logger.info("Stdout and file test")
    with logger.log_file.open("r") as f:
        logs = f.read()
    assert "Stdout and file test" in logs
    logger.logger.handlers.clear()
    logger.initialized = False


def test_action_logger_log_file_property(action_logger):
    assert isinstance(action_logger.log_file, Path)
    assert action_logger.log_file.exists()


def test_set_log_path_changes_action_logger_file(tmp_path):
    reset_logger_singletons()
    from xsam.logger import get_action_logger

    new_dir = tmp_path / "new_action_logs"
    set_log_path(new_dir)
    logger = get_action_logger()
    logger.info("Path change test")
    expected_log = new_dir / "action_log.log"
    assert expected_log.exists()
    with expected_log.open("r") as f:
        logs = f.read()
    assert "Path change test" in logs
    logger.logger.handlers.clear()
    logger.initialized = False


def test_file_logger_log_and_search(file_logger):
    file_path = "test_file.txt"
    file_logger.log_a_file(file_path)
    logs = file_logger.get_logs()
    assert len(logs) == 1
    log_entry = logs[0].strip().split(" | ")
    assert len(log_entry) == 3
    assert log_entry[2].endswith(file_path)
    retrieved_path = file_logger.search_logs(keyword=file_path, return_type="path")[-1]
    assert retrieved_path.name == file_path
    log_id = file_logger.search_logs(keyword=file_path, return_type="log_id")[-1]
    retrieved_path2 = file_logger.search_logs(keyword=log_id, return_type="path")[-1]
    assert str(retrieved_path2).endswith(file_path)


def test_file_logger_backup_and_clean(file_logger, tmp_path):
    reset_logger_singletons()
    file_path = tmp_path / "test_file.txt"
    file_path.touch()
    logger = FileLogger(tmp_path / "file_log")
    logger.log_a_file(str(file_path))
    backup_dir = tmp_path / "backup"
    logger.backup_logs(backup_dir)
    backup_files = list(backup_dir.glob("*.log"))
    assert len(backup_files) == 1
    logger.clean_logs()
    logs = logger.get_logs()
    assert len(logs) == 1  # file still exists, so log remains
    file_path.unlink()  # remove file
    logger.clean_logs()
    logs = logger.get_logs()
    assert len(logs) == 0


def test_set_log_path_changes_file_logger_file(tmp_path):
    reset_logger_singletons()
    new_dir = tmp_path / "new_file_logs"
    set_log_path(new_dir)
    from xsam.logger import get_file_logger

    logger = get_file_logger()
    test_file_path = new_dir / "test_file.txt"
    test_file_path.touch()
    logger.log_a_file(str(test_file_path))
    expected_log = new_dir / "file_log.log"
    assert expected_log.exists()
    with expected_log.open("r") as log_file:
        logs = log_file.readlines()
    assert any(str(test_file_path) in log for log in logs)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
