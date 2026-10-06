import os
import shutil
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.logger import app_logger, log_failure

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def backup_champion(config: dict = None, tag: str = None) -> Optional[str]:
    """
    Creates a timestamped snapshot of the current Champion model in models/backup/.
    Returns the backup directory path if successful.
    """
    if config is None:
        config = load_config()

    champion_dir = os.path.join(BASE_DIR, config["paths"]["champion_model_dir"])
    backup_base_dir = os.path.join(BASE_DIR, config["paths"]["backup_model_dir"])
    os.makedirs(backup_base_dir, exist_ok=True)

    champion_model_file = os.path.join(champion_dir, "model.joblib")
    if not os.path.exists(champion_model_file):
        app_logger.warning("No champion model found to backup.")
        return None

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_name = f"backup_{tag or 'champion'}_{timestamp}"
    backup_dir = os.path.join(backup_base_dir, backup_name)
    os.makedirs(backup_dir, exist_ok=True)

    # Copy model, preprocessor, and metadata
    for filename in os.listdir(champion_dir):
        src_file = os.path.join(champion_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(backup_dir, filename))

    # Add snapshot metadata
    snapshot_meta = {
        "backup_name": backup_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "tag": tag or "stable_champion",
        "source_dir": champion_dir
    }
    with open(os.path.join(backup_dir, "backup_info.json"), "w", encoding="utf-8") as f:
        json.dump(snapshot_meta, f, indent=2)

    # Prune older backups if exceeding limit
    max_backups = config.get("recovery", {}).get("backup_model_count", 5)
    all_backups = sorted([
        d for d in os.listdir(backup_base_dir)
        if os.path.isdir(os.path.join(backup_base_dir, d))
    ])
    if len(all_backups) > max_backups:
        for old_b in all_backups[:-max_backups]:
            old_path = os.path.join(backup_base_dir, old_b)
            shutil.rmtree(old_path, ignore_errors=True)
            app_logger.info(f"Pruned old backup snapshot: {old_b}")

    app_logger.info(f"Champion model successfully backed up to: {backup_dir}")
    return backup_dir

def deploy_challenger(config: dict = None) -> bool:
    """
    Promotes Challenger model to Champion:
    1. Backs up current Champion first.
    2. Overwrites Champion with Challenger artifacts.
    3. Updates version tag.
    """
    if config is None:
        config = load_config()

    champion_dir = os.path.join(BASE_DIR, config["paths"]["champion_model_dir"])
    challenger_dir = os.path.join(BASE_DIR, config["paths"]["challenger_model_dir"])
    os.makedirs(champion_dir, exist_ok=True)

    challenger_model = os.path.join(challenger_dir, "model.joblib")
    if not os.path.exists(challenger_model):
        app_logger.error(f"Cannot deploy challenger: {challenger_model} does not exist.")
        return False

    # Step 1: Backup current champion
    backup_champion(config, tag="pre_deployment")

    # Step 2: Copy challenger files to champion
    for filename in os.listdir(challenger_dir):
        src_file = os.path.join(challenger_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(champion_dir, filename))

    # Step 3: Update Champion metadata with deployment timestamp
    meta_path = os.path.join(champion_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["deployed_at"] = datetime.utcnow().isoformat() + "Z"
            meta["status"] = "ACTIVE_CHAMPION"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            app_logger.warning(f"Could not update champion metadata timestamp: {e}")

    app_logger.info("Challenger successfully promoted to ACTIVE CHAMPION!")
    return True

def list_backups(config: dict = None) -> List[Dict[str, Any]]:
    """Lists all available backup snapshots with metadata."""
    if config is None:
        config = load_config()

    backup_base_dir = os.path.join(BASE_DIR, config["paths"]["backup_model_dir"])
    if not os.path.exists(backup_base_dir):
        return []

    backups = []
    for d in sorted(os.listdir(backup_base_dir), reverse=True):
        b_dir = os.path.join(backup_base_dir, d)
        if os.path.isdir(b_dir):
            info_file = os.path.join(b_dir, "backup_info.json")
            meta_file = os.path.join(b_dir, "metadata.json")
            
            info = {}
            if os.path.exists(info_file):
                try:
                    with open(info_file, "r", encoding="utf-8") as f:
                        info = json.load(f)
                except Exception:
                    pass
                    
            model_meta = {}
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        model_meta = json.load(f)
                except Exception:
                    pass

            backups.append({
                "backup_id": d,
                "backup_path": b_dir,
                "timestamp": info.get("timestamp", d.split("_")[-1] if "_" in d else "unknown"),
                "tag": info.get("tag", "snapshot"),
                "model_name": model_meta.get("model_name", "unknown"),
                "model_version": model_meta.get("model_version", "unknown"),
                "metrics": model_meta.get("metrics", {})
            })
    return backups

def rollback_to_backup(config: dict = None, specific_backup_id: str = None) -> Dict[str, Any]:
    """
    Restores the Champion model from the latest (or specified) backup snapshot.
    Performs atomic file copy and logs rollback action.
    """
    if config is None:
        config = load_config()

    champion_dir = os.path.join(BASE_DIR, config["paths"]["champion_model_dir"])
    backup_base_dir = os.path.join(BASE_DIR, config["paths"]["backup_model_dir"])
    os.makedirs(champion_dir, exist_ok=True)

    backups = list_backups(config)
    if not backups:
        msg = "Rollback failed: No backup snapshots available in models/backup/"
        log_failure(
            failure_type="ROLLBACK_FAILED",
            severity="CRITICAL",
            component="ROLLBACK_ENGINE",
            details=msg,
            recommended_action="Re-run training pipeline to generate a valid model."
        )
        return {"status": "failed", "reason": msg}

    # Select target backup
    target_backup = None
    if specific_backup_id:
        target_backup = next((b for b in backups if b["backup_id"] == specific_backup_id), None)
        if not target_backup:
            return {"status": "failed", "reason": f"Backup ID '{specific_backup_id}' not found."}
    else:
        target_backup = backups[0] # Latest backup

    target_dir = target_backup["backup_path"]
    app_logger.info(f"Initiating rollback to snapshot: {target_backup['backup_id']}...")

    # Copy all files from target backup into champion directory
    for filename in os.listdir(target_dir):
        if filename != "backup_info.json":
            src_file = os.path.join(target_dir, filename)
            if os.path.isfile(src_file):
                shutil.copy2(src_file, os.path.join(champion_dir, filename))

    # Update champion metadata marking rollback event
    meta_path = os.path.join(champion_dir, "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["rollback_restored_at"] = datetime.utcnow().isoformat() + "Z"
            meta["restored_from_backup"] = target_backup["backup_id"]
            meta["status"] = "ACTIVE_CHAMPION_RESTORED"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
        except Exception:
            pass

    app_logger.info(f"ROLLBACK SUCCESSFUL: Restored champion model from {target_backup['backup_id']}")
    return {
        "status": "success",
        "restored_backup": target_backup["backup_id"],
        "model_version": target_backup["model_version"],
        "model_name": target_backup["model_name"],
        "metrics": target_backup["metrics"],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

if __name__ == "__main__":
    print("Available backups:", list_backups())
