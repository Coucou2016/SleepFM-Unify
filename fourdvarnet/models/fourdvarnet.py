"""4DVarNet: variational cost (Eq. 8) + unfolded gradient solver (Eq. 9)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from fourdvarnet.models.blocks import ConvFeatureNet, PriorUNet
from fourdvarnet.models.conv_lstm import ConvLSTMCell
from fourdvarnet.ops.physics import divergence, geostrophic_uv_from_ssh, gradient_xy


@dataclass
class FourDVarNetConfig:
    n_iter: int = 8
    n_features: int = 20
    lstm_hidden: int = 64
    lambda_obs: float = 1.0
    lambda_sst: float = 0.5
    lambda_prior: float = 0.1
    state_channels: int = 3  # SSH, u, v
    dx: float = 1.0
    dy: float = 1.0
    f_coriolis: float = 7.0e-5
    dx_m: float = 5550.0
    dy_m: float = 5550.0
    ssh_mean: float = 0.0
    ssh_std: float = 1.0
    u_mean: float = 0.0
    u_std: float = 1.0
    v_mean: float = 0.0
    v_std: float = 1.0
    residual_geo: bool = True
    alpha_uv_geo: float = 0.05


class FourDVarNet(nn.Module):
    """
    Multimodal 4DVarNet for SSH + SSC from altimetry and SST (Fablet et al., JAMES 2024).

    State x = (SSH, u, v) per time step, stacked as (B, 3*T, H, W).
    Observations y: DUACS-filled SSH + gappy tracks; u,v masked (never observed).
    SST z informs x via trainable G(x) ~ H(z) term in variational cost.
    """

    def __init__(self, cfg: FourDVarNetConfig | None = None, n_timesteps: int = 7):
        super().__init__()
        self.cfg = cfg or FourDVarNetConfig()
        self.n_timesteps = n_timesteps
        c = self.cfg
        in_ch = c.state_channels * n_timesteps

        self.phi = PriorUNet(in_ch)
        self.g_ssh = ConvFeatureNet(c.state_channels, c.n_features)
        self.h_sst = ConvFeatureNet(1, c.n_features)
        self.lstm = ConvLSTMCell(input_dim=in_ch, hidden_dim=c.lstm_hidden)
        self.step_scale = nn.Conv2d(c.lstm_hidden, in_ch, 1)
        nn.init.zeros_(self.step_scale.weight)
        nn.init.zeros_(self.step_scale.bias)

        self.lambda_obs = c.lambda_obs
        self.lambda_sst = c.lambda_sst
        self.lambda_prior = c.lambda_prior

    def _denorm_ssh(self, ssh: torch.Tensor) -> torch.Tensor:
        return ssh * self.cfg.ssh_std + self.cfg.ssh_mean

    def _norm_uv(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            (u - self.cfg.u_mean) / self.cfg.u_std,
            (v - self.cfg.v_mean) / self.cfg.v_std,
        )

    def _uv_from_ssh_phys(self, ssh_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ssh_phys = self._denorm_ssh(ssh_norm)
        u_g, v_g = geostrophic_uv_from_ssh(
            ssh_phys,
            f_coriolis=self.cfg.f_coriolis,
            dx=self.cfg.dx_m,
            dy=self.cfg.dy_m,
        )
        return self._norm_uv(u_g, v_g)

    def _split_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, 3*T, H, W) -> SSH, u, v each (B, T, H, W)."""
        b, _, h, w = x.shape
        t = self.n_timesteps
        x = x.view(b, t, 3, h, w)
        return x[:, :, 0], x[:, :, 1], x[:, :, 2]

    def _merge_state(self, ssh: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        b, t, h, w = ssh.shape
        stacked = torch.stack([ssh, u, v], dim=2)
        return stacked.reshape(b, t * 3, h, w)

    def variational_cost(
        self,
        x: torch.Tensor,
        y_obs: torch.Tensor,
        obs_mask: torch.Tensor,
        z_sst: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. 8: U = λ1||y-x||²_Ω + λ2||G(x)-H(z)||² + γ||x-Φ(x)||²."""
        diff_obs = (y_obs - x) * obs_mask
        term_obs = (diff_obs ** 2).mean()

        ssh, u, v = self._split_state(x)
        state_last = torch.cat([ssh[:, -1:], u[:, -1:], v[:, -1:]], dim=1)
        feat_x = self.g_ssh(state_last)
        feat_z = self.h_sst(z_sst)
        term_sst = ((feat_x - feat_z) ** 2).mean()

        term_prior = ((x - self.phi(x)) ** 2).mean()
        return (
            self.lambda_obs * term_obs
            + self.lambda_sst * term_sst
            + self.lambda_prior * term_prior
        )

    def forward(
        self,
        y_obs: torch.Tensor,
        obs_mask: torch.Tensor,
        z_sst: torch.Tensor,
        x_init: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = y_obs.clone() if x_init is None else x_init
        if obs_mask.sum() > 0:
            x = torch.where(obs_mask > 0, y_obs, x)

        lstm_state = None
        for _ in range(self.cfg.n_iter):
            x = x.detach().requires_grad_(True)
            cost = self.variational_cost(x, y_obs, obs_mask, z_sst)
            grad_x = torch.autograd.grad(cost, x, create_graph=self.training)[0]
            h, c = self.lstm(grad_x, lstm_state)
            lstm_state = (h, c)
            x = x - self.step_scale(h)

        ssh, u, v = self._split_state(x)
        ssh_last = ssh[:, -1:]
        u_last, v_last = u[:, -1:], v[:, -1:]
        if self.cfg.residual_geo:
            u_geo, v_geo = self._uv_from_ssh_phys(ssh_last)
            u_last = self.cfg.alpha_uv_geo * u_last + u_geo
            v_last = self.cfg.alpha_uv_geo * v_last + v_geo

        return {
            "x": x,
            "ssh": ssh,
            "u": u,
            "v": v,
            "ssh_last": ssh_last,
            "u_last": u_last,
            "v_last": v_last,
        }

    def training_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Supervised losses Eqs. 10–14 (SSH, grad SSH, u/v, divergence, Phi regularization)."""
        w = weights or {
            "ssh": 50.0,
            "grad_ssh": 1000.0,
            "uv": 50.0,
            "uv_geo": 25.0,
            "div": 1000.0,
            "phi": 1.0,
        }
        ssh_p, ssh_t = outputs["ssh_last"], target["ssh"]
        u_p, v_p = outputs["u_last"], outputs["v_last"]
        u_t, v_t = target["u"], target["v"]

        l_ssh = ((ssh_p - ssh_t) ** 2).mean()
        gx_p, gy_p = gradient_xy(ssh_p, self.cfg.dx, self.cfg.dy)
        gx_t, gy_t = gradient_xy(ssh_t, self.cfg.dx, self.cfg.dy)
        l_grad = ((gx_p - gx_t) ** 2 + (gy_p - gy_t) ** 2).mean()
        l_uv = ((u_p - u_t) ** 2 + (v_p - v_t) ** 2).mean()

        u_geo_p, v_geo_p = self._uv_from_ssh_phys(ssh_p)
        u_geo_t, v_geo_t = self._uv_from_ssh_phys(ssh_t)
        l_uv_geo = ((u_geo_p - u_geo_t) ** 2 + (v_geo_p - v_geo_t) ** 2).mean()

        # Physical div loss uses 1e4 in the official repo; z-scored fields omit that factor.
        div_scale = 1.0
        l_div = (
            div_scale
            * (
                divergence(u_p, v_p, self.cfg.dx, self.cfg.dy)
                - divergence(u_t, v_t, self.cfg.dx, self.cfg.dy)
            )
            ** 2
        ).mean()

        x_true = self._merge_state(target["ssh_seq"], target["u_seq"], target["v_seq"])
        x_hat = outputs["x"]
        l_phi = ((x_true - self.phi(x_true)) ** 2 + (x_hat - self.phi(x_hat)) ** 2).mean()

        total = (
            w["ssh"] * l_ssh
            + w["grad_ssh"] * l_grad
            + w["uv"] * l_uv
            + w.get("uv_geo", 25.0) * l_uv_geo
            + w["div"] * l_div
            + w["phi"] * l_phi
        )
        stats = {
            "loss": float(total.detach()),
            "l_ssh": float(l_ssh.detach()),
            "l_grad_ssh": float(l_grad.detach()),
            "l_uv": float(l_uv.detach()),
            "l_uv_geo": float(l_uv_geo.detach()),
            "l_div": float(l_div.detach()),
            "l_phi": float(l_phi.detach()),
        }
        return total, stats
