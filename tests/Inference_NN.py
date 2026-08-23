import numpy as np
import torch
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from channel_model import generate_channel, array_response_far_field, array_response_near_field
from sinr_sumrate import sum_rate
from network import MultiuserAnalogPrecoderNet
from loss import general_mse_objective, concentrated_mse_loss, kkt_digital_precoder_torch
from channel_model_los import generate_channel_los
M, K, L, N_RF,T = 128, 4, 1, 4, 0
fc, c = 100e9, 3e8
WL = c / fc
d = WL / 2
P_t = 1.0
noise_power = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiuserAnalogPrecoderNet(
    M=M, K=K, T=T, N_RF=N_RF, mlp_dims=[512, 256, 128]
).to(device)  
MODEL_PATH = PROJECT_ROOT / "models" / "Best_comm_model_L1.pt"
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def compute_1d_beam_pattern(
    F_RF: np.ndarray,
    F_BB: np.ndarray,
    M: int,
    d: float,
    wavelength: float,
    n_theta: int = 1000,
    r: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:

    theta_grid = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
    X = F_RF @ F_BB  # (M, K) -- the actual transmitted spatial pattern

    gain = np.empty(n_theta)
    for i, theta in enumerate(theta_grid):
        if r is None:
            a = array_response_far_field(theta, M, d, wavelength)
        else:
            a = array_response_near_field(theta, r, M, d, wavelength)
        # power radiated toward this (theta, r) cut, summed over all K
        # data streams -- i.e. ||a^H X||_2^2
        gain[i] = np.sum(np.abs(a.conj() @ X) ** 2)

    gain_dB = 10 * np.log10(gain / (gain.max() + 1e-12) + 1e-12)
    theta_deg = np.rad2deg(theta_grid)
    return theta_deg, gain_dB


def run_demo(rng: np.random.Generator | None = None, beam_pattern_r: float | None = None) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    H, theta_users_deg, r_users = generate_channel_los(
        K=K, M=M, d=d, wavelength=WL, r_min=5.0, r_max=80.0, rng=rng
    )
    H_torch = torch.as_tensor(H[np.newaxis], dtype=torch.cfloat, device=device)  # (1, M, K)

    with torch.no_grad():
        F_RF_nn_torch = model(H_torch)                                          # (1, M, N_RF)
        F_BB_nn_torch, _, _ = kkt_digital_precoder_torch(
            H_torch, F_RF_nn_torch, noise_power, P_t
        )                                                                        # (1, N_RF, K)
        mse_concentrated = concentrated_mse_loss(
            H_torch, F_RF_nn_torch, noise_power, P_t
        ).item()

    F_RF_nn = F_RF_nn_torch.squeeze(0).cpu().numpy()
    F_BB_nn = F_BB_nn_torch.squeeze(0).cpu().numpy()

    rate = sum_rate(H, F_RF_nn, F_BB_nn, noise_power)
    mse_general = general_mse_objective(
        H=H, F_RF=F_RF_nn, F_BB=F_BB_nn, noise_power=noise_power, P_t=P_t, beta=None
    )
    theta_deg, gain_dB = compute_1d_beam_pattern(
        F_RF_nn, F_BB_nn, M, d, WL, r=beam_pattern_r
    )
    return {
        "H": H,
        "theta_users_deg": theta_users_deg,
        "r_users": r_users,
        "F_RF": F_RF_nn,
        "F_BB": F_BB_nn,
        "sum_rate": rate,
        "mse_general": mse_general,
        "mse_concentrated": mse_concentrated,
        "beam_pattern_theta_deg": theta_deg,
        "beam_pattern_gain_dB": gain_dB,
    }


if __name__ == "__main__":
    result = run_demo()

    print(f"Sum rate         : {result['sum_rate']:.4f} bps/Hz")
    print(f"General MSE      : {result['mse_general']:.6f}")
    print(f"Concentrated MSE : {result['mse_concentrated']:.6f}")
    print(f"User angles (deg): {np.round(result['theta_users_deg'], 2)}")
    print(f"User ranges (m)  : {np.round(result['r_users'], 2)}")
    peak_theta = result["beam_pattern_theta_deg"][np.argmax(result["beam_pattern_gain_dB"])]
    print(f"Beam pattern peak at {peak_theta:.2f} deg")
    out_dir = PROJECT_ROOT / "Outputs_L1"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(result["beam_pattern_theta_deg"], result["beam_pattern_gain_dB"])
    for th in result["theta_users_deg"]:
        plt.axvline(th, color="r", linestyle="--", alpha=0.4)
    plt.xlabel("Angle (deg)")
    plt.ylabel("Normalized gain (dB)")
    plt.title("1-D beam pattern (far-field cut) -- dashed lines mark true user angles")
    plt.ylim(-40, 2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = out_dir / "beam_pattern_ex3.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved beam pattern plot to {save_path}")
