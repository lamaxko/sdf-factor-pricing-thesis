import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class MacroStateLSTM(nn.Module):
    def __init__(self, input_size=None, num_states=4, hidden_size=32, seq_len=12):
        super().__init__()
        self.seq_len = seq_len
        self.num_states = num_states
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.lstm = None
        self.linear = None

    def _init_layers(self, actual_input_size):
        self.input_size = actual_input_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.num_states)

    def forward(self, x):
        """
        x: Tensor of shape (batch, seq_len, input_size)
        returns: Tensor of shape (batch, seq_len, num_states)
        """
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out)

    def extract_from_dataframe(self, df, min_coverage=0.7, min_active_features=20, smooth_window=6):
        """
        Process a DataFrame of macro features into latent state time series.
        Returns a (T, num_states) DataFrame.
        """
        df = df.copy()
        coverage = df.notna().mean()
        retained_cols = coverage[coverage >= min_coverage].index.tolist()
        df = df[retained_cols].ffill()

        all_cols = df.columns
        num_features = len(all_cols)

        self._init_layers(num_features)

        sequences = []
        valid_indices = []

        for i in range(self.seq_len, len(df)):
            window = df.iloc[i - self.seq_len:i]
            valid_cols = [col for col in all_cols if not window[col].isnull().any()]

            if len(valid_cols) >= min_active_features:
                tensor = torch.tensor(window[valid_cols].values, dtype=torch.float32)
                padded = torch.zeros((self.seq_len, num_features), dtype=torch.float32)
                for j, col in enumerate(valid_cols):
                    col_idx = all_cols.get_loc(col)
                    padded[:, col_idx] = tensor[:, j]
                sequences.append(padded)
                valid_indices.append(df.index[i])

        if not sequences:
            raise ValueError("No valid windows found. Adjust filtering thresholds.")

        I_t = torch.stack(sequences)

        self.eval()
        with torch.no_grad():
            h_t = self.forward(I_t)

        h_np = h_t[:, -1, :].cpu().numpy()
        h_df = pd.DataFrame(h_np, index=pd.to_datetime(valid_indices), columns=[f"h{i}" for i in range(h_np.shape[1])])
        h_df = h_df.rolling(window=smooth_window, min_periods=1, center=True).mean()

        return h_df
