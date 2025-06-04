import torch
import pandas as pd
from dataclasses import dataclass

@dataclass
class ModelInputs:
    x_omega: torch.Tensor
    x_g: torch.Tensor
    y: torch.Tensor
    panel: pd.DataFrame

class TensorPreprocessor:
    def __init__(self, macro_states_sdf: pd.DataFrame, macro_states_cond: pd.DataFrame, panel_data: pd.DataFrame):
        """
        Initializes the preprocessor with macro latent states and asset-level panel data.
        """
        self.h_t = macro_states_sdf.copy()
        self.h_t_g = macro_states_cond.copy()
        self.I_ti = panel_data.copy()
        self.merged = None
        self.x_omega_cols = []
        self.x_g_cols = []
        self.id_cols = ["date", "id_stock", "ExcessRet"]

    def preprocess(self):
        """
        Merges, aligns, cleans, and prepares tensors for training the SDF GAN model.
        Returns:
            x_omega: Tensor for SDF network input
            x_g: Tensor for conditional network input
            y: Tensor of excess returns
        """
        # Ensure datetime alignment
        self.I_ti["date"] = pd.to_datetime(self.I_ti["date"])
        self.h_t.index = pd.to_datetime(self.h_t.index)
        self.h_t_g.index = pd.to_datetime(self.h_t_g.index)

        # Merge latent states into the panel
        merged = self.I_ti.merge(self.h_t, left_on="date", right_index=True, how="inner")
        merged = merged.merge(self.h_t_g, left_on="date", right_index=True, suffixes=('', '_g'))

        # Define features
        all_features = [col for col in merged.columns if col not in self.id_cols]
        self.x_omega_cols = [col for col in all_features if not col.endswith("_g")]
        self.x_g_cols = [col for col in all_features if col.endswith("_g") or col not in self.h_t.columns]

        # Log NaNs before dropping
        print(f"Before dropna(): merged shape = {merged.shape}")

        # Drop NaNs
        merged = merged.dropna().reset_index(drop=True)
        self.merged = merged  # store for inspection if needed

        # Log diagnostics after drop
        print(f"After dropna(): merged shape = {merged.shape}")

        # Convert to tensors
        x_omega = torch.tensor(merged[self.x_omega_cols].values, dtype=torch.float32)
        x_g = torch.tensor(merged[self.x_g_cols].values, dtype=torch.float32)
        y = torch.tensor(merged["ExcessRet"].values, dtype=torch.float32)

        # Sanity stats
        print("==== Tensor Summary ====")
        print(f"x_omega shape: {x_omega.shape}, mean: {x_omega.mean():.4f}, std: {x_omega.std():.4f}")
        print(f"x_g     shape: {x_g.shape}, mean: {x_g.mean():.4f}, std: {x_g.std():.4f}")
        print(f"y        shape: {y.shape}, mean: {y.mean():.4f}, std: {y.std():.4f}")

        return ModelInputs(x_omega=x_omega, x_g=x_g, y=y, panel=self.merged)
