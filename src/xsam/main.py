from xsam.logger import ActionLogger


def main():
    print("Welcome to the XSAM package!")

    action_logger = ActionLogger()
    action_logger.info("Starting main function")

    run_tests = input("Would you like to run all tests? (y/n): ")
    if run_tests.lower() == "y":
        action_logger.info("Running all tests")
        import pytest

        pytest.main()
    else:
        action_logger.info("Skipping tests")


if __name__ == "__main__":
    main()
