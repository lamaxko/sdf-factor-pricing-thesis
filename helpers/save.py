import os
import pandas as pd

def save_gan(panel, training_log, save_dir, syear, eyear):
    os.makedirs(os.path.join(save_dir, "panel"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "log"), exist_ok=True)

    panel_filename = f"panel_{syear}_{eyear}.csv"
    log_filename = f"training_log_{syear}_{eyear}.csv"

    panel_path = os.path.join(save_dir, "panel", panel_filename)
    log_path = os.path.join(save_dir, "log", log_filename)

    panel.to_csv(panel_path, index=False)
    training_log.to_csv(log_path, index=False)

    print(f"Saved panel to {panel_path}")
    print(f"Saved training log to {log_path}")
    return
