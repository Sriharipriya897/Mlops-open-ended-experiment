import os
import json
import threading
import joblib
import yaml
from typing import Tuple, Dict, Any, Optional

from src.logger import app_logger, log_failure
from src.rollback import rollback_to_backup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

class ModelManager:
    """Thread-safe model loader and failover manager."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance.model = None
                cls._instance.preprocessor = None
                cls._instance.metadata = {}
                cls._instance.is_loaded = False
        return cls._instance

    def load_model(self, force_reload: bool = False, enable_auto_failover: bool = True):
        with self._lock:
            if self.is_loaded and not force_reload:
                return self.model, self.preprocessor, self.metadata

            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            champion_dir = os.path.join(BASE_DIR, config["paths"]["champion_model_dir"])
            model_path = os.path.join(champion_dir, "model.joblib")
            prep_path = os.path.join(champion_dir, "preprocessor.joblib")
            meta_path = os.path.join(champion_dir, "metadata.json")

            try:
                if not os.path.exists(model_path) or not os.path.exists(prep_path):
                    raise FileNotFoundError(f"Missing champion artifacts in {champion_dir}")

                self.model = joblib.load(model_path)
                self.preprocessor = joblib.load(prep_path)

                if os.path.exists(meta_path):
                    with open(meta_path, "r", encoding="utf-8") as f:
                        self.metadata = json.load(f)
                else:
                    self.metadata = {
                        "model_name": "ChampionModel",
                        "model_version": "v1.0-default",
                        "status": "ACTIVE_CHAMPION",
                        "metrics": {},
                        "feature_names": []
                    }

                self.is_loaded = True
                app_logger.info(f"ModelManager: Loaded champion model {self.metadata.get('model_version')}")
                return self.model, self.preprocessor, self.metadata

            except Exception as e:
                log_failure(
                    failure_type="MODEL_LOAD_FAILURE",
                    severity="CRITICAL",
                    component="MODEL_MANAGER",
                    details=f"Failed loading champion model: {str(e)}",
                    recommended_action="Initiating failover recovery from latest backup."
                )

                if enable_auto_failover:
                    app_logger.warning("Attempting automated failover to last known-good backup...")
                    res = rollback_to_backup(config)
                    if res.get("status") == "success":
                        # Try reloading after rollback
                        self.model = joblib.load(model_path)
                        self.preprocessor = joblib.load(prep_path)
                        if os.path.exists(meta_path):
                            with open(meta_path, "r", encoding="utf-8") as f:
                                self.metadata = json.load(f)
                        self.is_loaded = True
                        app_logger.info("Failover recovery SUCCESSFUL. Backup restored.")
                        return self.model, self.preprocessor, self.metadata

                self.is_loaded = False
                raise RuntimeError(f"Model loading and failover failed: {str(e)}")

model_manager = ModelManager()
