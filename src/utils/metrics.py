import time
import logging
import json
import os

class MetricsCollector:
    def __init__(self, log_file="data/pipeline_metrics.json"):
        self.metrics = {}
        self.log_file = log_file
        self.logger = logging.getLogger("MetricsCollector")
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def start_timer(self, name):
        self.metrics[f"{name}_start_time"] = time.time()

    def stop_timer(self, name):
        start_time = self.metrics.get(f"{name}_start_time")
        if start_time:
            duration = time.time() - start_time
            self.metrics[f"{name}_duration_sec"] = round(duration, 4)
            del self.metrics[f"{name}_start_time"]
            return duration
        return 0

    def set_metric(self, name, value):
        self.metrics[name] = value

    def increment_metric(self, name, count=1):
        self.metrics[name] = self.metrics.get(name, 0) + count

    def get_summary(self):
        return self.metrics

    def save_to_file(self):
        try:
            # Append if file exists? Or overwrite for each run? 
            # Usually, one run per file or nested by timestamp is better.
            # For simplicity, we'll write the latest run with a timestamp.
            run_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": self.metrics
            }
            
            # For this simple case, let's append as JSONL to keep history
            with open(self.log_file, "a") as f:
                f.write(json.dumps(run_data) + "\n")
            
            self.logger.info(f"Metrics saved to {self.log_file}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def log_summary(self):
        self.logger.info("=== METRICS SUMMARY ===")
        for k, v in self.metrics.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("=======================")
