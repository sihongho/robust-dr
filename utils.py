import os
import logging
from datetime import datetime
import json
import subprocess


def setup_logging_and_dirs():
    """Set up logging and create directories for experiment outputs."""
    # Create a timestamped directory for the experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"./experiments/{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(experiment_dir, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Experiment directory created: {experiment_dir}")
    return experiment_dir


def save_config(experiment_dir, config):
    """Save configuration and environment data to the experiment directory."""
    config_path = os.path.join(experiment_dir, "config.json")
    
    # Save config as JSON
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    logging.info(f"Configuration saved to {config_path}")

