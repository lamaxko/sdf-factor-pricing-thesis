import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
dir = r"out/log/"

data = {}

def plot_the_fucks(eyear):
    path = fr"{dir}training_log_2000_{eyear}.csv"


# Define color palette
    COLOR_OMEGA = "#328cc1"
    COLOR_G = "#c94c4c"
    COLOR_LOSS = "#4c956c"      # Elegant Green

    def apply_default_style():
        plt.style.use("default")
        plt.rcParams.update({
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "black",
            "axes.linewidth": 1,
            "axes.labelweight": "bold",
            "axes.grid": True,
            "grid.color": "#cccccc",
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "axes.titleweight": "bold"
        })

# File paths
    input_path = path
    output_dir = fr"{dir}plots/"
    os.makedirs(output_dir, exist_ok=True)

# Load data
    df = pd.read_csv(input_path)
    df = df.apply(pd.to_numeric, errors='coerce')

# Apply consistent visual style
    apply_default_style()

# --- Plot 1: Omega vs G Mean ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.lineplot(x=df.index, y=df["omega_mean"], color=COLOR_OMEGA, linewidth=2, ax=ax1)
    ax1.set_ylabel("Omega Mean", color=COLOR_OMEGA, fontsize=12, weight="bold")
    ax1.tick_params(axis='y', labelcolor=COLOR_OMEGA)
    ax1.set_xlabel("Epoch", fontsize=12, weight="bold", color="#0b3c5d")

    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)  # <-- Add this line to show the right y-axis line
    sns.lineplot(x=df.index, y=df["g_mean"], color=COLOR_G, linewidth=2, ax=ax2)
    ax2.set_ylabel("g Mean", color=COLOR_G, fontsize=12, weight="bold")
    ax2.tick_params(axis='y', labelcolor=COLOR_G)
    ax1.set_title(f"Omega vs g Mean Over Epochs Training Period: 2000-{eyear}", fontsize=16, weight="bold", color="#0b3c5d")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"omega_vs_g_mean_{eyear}.png"), dpi=300)
    plt.close()

# --- Plot 2: Loss over Epochs ---
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.lineplot(x=df.index, y=df["loss_omega"], color=COLOR_LOSS, linewidth=2)
    ax.set_title(f"Loss (Omega) Over Epochs Training Period: 2000-{eyear}", fontsize=16, weight="bold", color="#0b3c5d")
    ax.set_xlabel("Epoch", fontsize=12, weight="bold", color="#0b3c5d")
    ax.set_ylabel("Loss", fontsize=12, weight="bold", color=COLOR_LOSS)
    ax.tick_params(axis='y', labelcolor=COLOR_LOSS)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"loss_omega_{eyear}.png"), dpi=300)
    plt.close()

# --- Combined Plot: Loss (top) and Omega vs G (bottom) ---
    fig, (ax_loss, ax1) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Top plot: Loss over Epochs
    sns.lineplot(x=df.index, y=df["loss_omega"], color=COLOR_LOSS, linewidth=2, ax=ax_loss)
    ax_loss.set_title(f"Loss (Omega) Over Epochs Training Period: 2000-{eyear}", fontsize=16, weight="bold", color="#0b3c5d")
    ax_loss.set_ylabel("Loss", fontsize=12, weight="bold", color=COLOR_LOSS)
    ax_loss.tick_params(axis='y', labelcolor=COLOR_LOSS)

# Bottom plot: Omega vs G Mean
    sns.lineplot(x=df.index, y=df["omega_mean"], color=COLOR_OMEGA, linewidth=2, ax=ax1)
    ax1.set_ylabel("Omega Mean", color=COLOR_OMEGA, fontsize=12, weight="bold")
    ax1.tick_params(axis='y', labelcolor=COLOR_OMEGA)
    ax1.set_xlabel("Epoch", fontsize=12, weight="bold", color="#0b3c5d")

# Add second y-axis for g_mean
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    sns.lineplot(x=df.index, y=df["g_mean"], color=COLOR_G, linewidth=2, ax=ax2)
    ax2.set_ylabel("g Mean", color=COLOR_G, fontsize=12, weight="bold")
    ax2.tick_params(axis='y', labelcolor=COLOR_G)

# Title for the bottom subplot
    ax1.set_title(f"Omega vs g Mean Over Epochs Training Period: 2000-{eyear}", fontsize=16, weight="bold", color="#0b3c5d")

# Save combined plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"combined_loss_omega_g_{eyear}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    for eyear in range(2006, 2024+1):
        plot_the_fucks(eyear)
