import os
import sys
import traceback

# Ensure project root is in path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    log_path = "/tmp/test_results.txt"
    with open(log_path, "w") as log_file:
        log_file.write("=== AI Security Ecosystem Unit Tests Execution ===\n")
        log_file.write(f"Timestamp: {os.popen('date').read().strip()}\n\n")
        
        # Test 1: Configuration Loading
        log_file.write("Test 1: Loading Configuration Files... ")
        try:
            from tests.test_config import test_load_configurations
            test_load_configurations()
            log_file.write("PASSED\n")
        except Exception as e:
            log_file.write(f"FAILED\n{traceback.format_exc()}\n")
            
        # Test 2: Tracker Manager History
        log_file.write("Test 2: Tracker Manager History... ")
        try:
            from tests.test_pipeline import test_tracker_manager_history
            test_tracker_manager_history()
            log_file.write("PASSED\n")
        except Exception as e:
            log_file.write(f"FAILED\n{traceback.format_exc()}\n")
            
        # Test 3: Line Crossing Logic
        log_file.write("Test 3: Line Crossing Logic... ")
        try:
            from tests.test_pipeline import test_line_crossing_logic
            test_line_crossing_logic()
            log_file.write("PASSED\n")
        except Exception as e:
            log_file.write(f"FAILED\n{traceback.format_exc()}\n")
            
        # Test 4: Zone Analytics Logic
        log_file.write("Test 4: Zone Analytics Logic... ")
        try:
            from tests.test_pipeline import test_zone_analytics_logic
            test_zone_analytics_logic()
            log_file.write("PASSED\n")
        except Exception as e:
            log_file.write(f"FAILED\n{traceback.format_exc()}\n")
            
        log_file.write("\n=== All Tests Processed ===\n")
        
    print(f"Tests finished. Results written to: {log_path}")

if __name__ == "__main__":
    main()
