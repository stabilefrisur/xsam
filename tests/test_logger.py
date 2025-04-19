import pytest

from xsam.logger import set_log_path, FileLogger


@pytest.fixture
def logger(tmp_path):
    log_file = tmp_path / "test_log"
    return FileLogger(log_file)


def test_log_file_path(logger):
    file_path = "test_file.txt"
    logger.log_file_path(file_path)
    logs = logger.get_logs()
    assert len(logs) == 1
    log_entry = logs[0].strip().split(" | ")
    assert len(log_entry) == 3
    assert log_entry[2].endswith(file_path)


def test_get_file_path_by_log_id(logger):
    file_path = "test_file.txt"
    logger.log_file_path(file_path)
    logs = logger.get_logs()
    log_id = logs[0].strip().split(" | ")[0]
    retrieved_path = logger.get_file_path(log_id=log_id)
    assert retrieved_path.endswith(file_path)


def test_get_file_path_by_file_name(logger):
    file_path = "test_file.txt"
    logger.log_file_path(file_path)
    retrieved_path = logger.get_file_path(file_name="test_file.txt")
    assert retrieved_path.endswith(file_path)


def test_get_logs(logger):
    file_path = "test_file.txt"
    logger.log_file_path(file_path)
    logs = logger.get_logs()
    assert len(logs) == 1
    assert logs[0].strip().endswith(file_path)


def test_search_logs(logger):
    file_path = "test_file.txt"
    logger.log_file_path(file_path)
    search_results = logger.search_logs("test_file")
    assert len(search_results) == 1
    assert search_results[0].strip().endswith(file_path)


def test_backup_logs(logger, tmp_path):
    file_path = "test_file.txt"
    logger.log_file_path(file_path)
    backup_dir = tmp_path / "backup"
    logger.backup_logs(backup_dir)
    backup_files = list(backup_dir.glob("*.log"))
    assert len(backup_files) == 1


def test_clean_logs(logger):
    file_path = "test_file.txt"
    logger.log_file_path(file_path)
    logger.clean_logs()
    logs = logger.get_logs()
    assert len(logs) == 0


def test_set_log_path(tmp_path):
    # Arrange: Create a temporary log path
    new_log_path = tmp_path / "custom_log.log"

    # Act: Call set_log_path with the new path
    set_log_path(new_log_path)

    # Assert: Verify that the FileLogger is using the new path
    file_logger = FileLogger(new_log_path)
    assert file_logger.log_file == new_log_path.with_suffix(".log")

    # Log a new file path
    test_file_path = tmp_path / "test_file.txt"
    test_file_path.touch()  # Create the test file
    file_logger.log_file_path(str(test_file_path))

    # Assert: Verify that the log file is created
    assert new_log_path.with_suffix(".log").exists()

    # Assert: Verify that the new file path is logged
    with new_log_path.with_suffix(".log").open("r") as log_file:
        logs = log_file.readlines()
    assert any(str(test_file_path) in log for log in logs)


if __name__ == "__main__":
    pytest.main()
