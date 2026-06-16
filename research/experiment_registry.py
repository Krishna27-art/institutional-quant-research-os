import json
import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

class ExperimentRegistry:
    """
    Tracks experiment lineage for institutional reproducibility.
    Provides persistence and strict schema for logging research.
    """
    def __init__(self, db_path: str = "research/experiment_registry.json"):
        self.db_path = Path(db_path)
        self._ensure_db()
        self.experiments = self._load()

    def _ensure_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            with open(self.db_path, "w") as f:
                json.dump({}, f)

    def _load(self) -> Dict[str, Any]:
        try:
            with open(self.db_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _save(self):
        with open(self.db_path, "w") as f:
            json.dump(self.experiments, f, indent=4)

    def register_experiment(
        self,
        hypothesis: str,
        dataset_version: str,
        feature_version: str,
        code_commit: str,
        author: str = "quant"
    ) -> str:
        """Register a new experiment prior to execution."""
        exp_id = f"EXP-{datetime.datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        
        self.experiments[exp_id] = {
            "hypothesis": hypothesis,
            "dataset_version": dataset_version,
            "feature_version": feature_version,
            "code_commit": code_commit,
            "author": author,
            "status": "Running",
            "created_at": datetime.datetime.now().isoformat(),
            "results": None,
            "promotion_status": "Draft" # Draft, Promoted, Rejected, Retired
        }
        self._save()
        return exp_id

    def log_results(self, exp_id: str, results: Dict[str, float]):
        """Log the results (e.g. Sharpe, Drawdown, CAGR) of the experiment."""
        if exp_id not in self.experiments:
            raise KeyError(f"Experiment {exp_id} not found.")
        
        self.experiments[exp_id]["results"] = results
        self.experiments[exp_id]["status"] = "Completed"
        self._save()

    def update_promotion_status(self, exp_id: str, status: str, notes: Optional[str] = None):
        """Update promotion status: Draft, Promoted, Rejected, Retired."""
        valid_statuses = ["Draft", "Promoted", "Rejected", "Retired"]
        if status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of {valid_statuses}")
            
        if exp_id not in self.experiments:
            raise KeyError(f"Experiment {exp_id} not found.")

        self.experiments[exp_id]["promotion_status"] = status
        if notes:
            self.experiments[exp_id]["promotion_notes"] = notes
        self._save()

    def get_experiment(self, exp_id: str) -> Dict[str, Any]:
        return self.experiments.get(exp_id, {})
    
    def list_promoted_experiments(self) -> Dict[str, Any]:
        return {k: v for k, v in self.experiments.items() if v["promotion_status"] == "Promoted"}
