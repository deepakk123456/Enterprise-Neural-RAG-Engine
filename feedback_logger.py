import os
import json
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
LOG_PATH = os.path.join(BASE_DIR, "telemetry_feedback.json")

class FeedbackLogger:
    @staticmethod
    def log_alignment_signal(query: str, response: str, rating: str):
        payload = {
            "epoch_timestamp": time.time(),
            "query_target": query,
            "synthesized_output": response,
            "alignment_signal": rating
        }
        data = []
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH, "r") as f: data = json.load(f)
            except Exception: data = []
        data.append(payload)
        with open(LOG_PATH, "w") as f: json.dump(data, indent=4, ensure_ascii=False)
