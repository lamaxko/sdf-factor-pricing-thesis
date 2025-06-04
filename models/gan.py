import torch
import pandas as pd
from models.ffn import FeedForwardNet

class SDFGan:
    def __init__(self, input_dim_omega, input_dim_g, hidden_dim=64, lr=1e-3):
        self.sdf_net = FeedForwardNet(input_dim=input_dim_omega, hidden_dim=hidden_dim, output_dim=1)
        self.cond_net = FeedForwardNet(input_dim=input_dim_g, hidden_dim=hidden_dim, output_dim=1)

        self._init_weights(self.sdf_net)
        self._init_weights(self.cond_net)

        self.opt_sdf = torch.optim.Adam(self.sdf_net.parameters(), lr=lr)
        self.opt_cond = torch.optim.Adam(self.cond_net.parameters(), lr=lr)

        self.training_log = []

    def _init_weights(self, model):
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0.1)

    def fit(self, x_omega, x_g, y, n_epochs=100, inner_steps=5):
        early_stop_counter = 0
        for epoch in range(n_epochs):
            for _ in range(inner_steps):
                omega = self.sdf_net(x_omega).squeeze()
                g = self.cond_net(x_g).squeeze().clamp(-5, 5)
                M = 1 - (omega * y).sum()
                moment_term = M * y * g
                loss_g = torch.mean(moment_term) ** 2

                self.opt_cond.zero_grad()
                (-loss_g).backward()
                self.opt_cond.step()

            for _ in range(inner_steps):
                omega = self.sdf_net(x_omega).squeeze()
                g = self.cond_net(x_g).squeeze().clamp(-5, 5)
                M = 1 - (omega * y).sum()
                moment_term = M * y * g
                loss_omega = torch.mean(moment_term) ** 2

                self.opt_sdf.zero_grad()
                loss_omega.backward()
                self.opt_sdf.step()

            # Logging
            stats = {
                "epoch": epoch,
                "loss_omega": loss_omega.item(),
                "M": M.item(),
                "sum_omega_y": (omega * y).sum().item(),
                "omega_mean": omega.mean().item(),
                "g_mean": g.mean().item()
            }
            self.training_log.append(stats)

            print(f"Epoch {epoch:03d} | Loss: {loss_omega.item():.6f}")
            print(f"  SDF Error (M):        {M.item():.6f}")
            print(f"  Sum(omega * R):             {(omega * y).sum().item():.6f}")
            print(f"  omega mean (SDF weights): {omega.mean().item():.4f}")
            print(f"  g mean (test fn):     {g.mean().item():.4f}")

            if loss_omega.item() < 1e-4 and abs(M.item()) < 0.01:
                early_stop_counter += 1
                if early_stop_counter >= 5:
                    print(f"[Early Stop] Conditions met at epoch {epoch:03d}")
                    break
            else:
                early_stop_counter = 0

    def extract_outputs(self, x_omega, x_g):
        self.sdf_net.eval()
        self.cond_net.eval()
        with torch.no_grad():
            omega = self.sdf_net(x_omega).squeeze().cpu().numpy()
            g = self.cond_net(x_g).squeeze().cpu().numpy()
        return omega, g

    def get_training_log(self):
        return pd.DataFrame(self.training_log)
