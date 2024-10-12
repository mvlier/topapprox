# conftest.py
'''
Configurations file for pytest package.
This functions determine the behaviour of pytest.
'''

import pytest
import time
from datetime import datetime

# Store all the test results
test_results = []
start_time = None

@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    # Capture the start time of the test session
    global start_time
    start_time = datetime.now().strftime("%Y-%m-%d, at %H:%M:%S")

def pytest_runtest_setup(item):
    # Mark the start time for each test (if needed later)
    item.start_time = time.time()

def pytest_runtest_teardown(item, nextitem):
    # Capture the duration for each test
    duration = time.time() - item.start_time
    test_results.append(f"{item.name} took {duration:.4f} seconds")

@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    global start_time, test_results
    
    # Log the test session date/time and all test results
    log_entry = f"\n{start_time}\n" + "\n".join(test_results) + "\n"

    # Read the existing log file (if it exists)
    log_file = './tests/test_log.txt'
    try:
        with open(log_file, 'r') as f:
            existing_content = f.read()
    except FileNotFoundError:
        existing_content = ""

    # Prepend the new log entry to the file
    with open(log_file, 'w') as f:
        f.write(log_entry + existing_content)
    
    # Reset test results for the next test run
    test_results = []

def pytest_runtest_logreport(report):
    """Hook that runs when each test is logged."""
    if report.when == "call" and report.passed:
        print(f"\nTest {report.nodeid} passed with:")
        print(f"  - Duration: {report.duration:.4f}s")
        print(f"  - Outcome: {report.outcome}")

