
import numpy as np
import torch

def kkt_digital_precoder(
    H: np.ndarray, F_RF: np.ndarray, noise_power: float, P_t: float
) -> tuple[np.ndarray, np.ndarray, float]:

    K = H.shape[1]
    N_RF = F_RF.shape[1]
    lambda_lag = K * noise_power / P_t
    FH_H = F_RF.conj().T @ H  # shape (N_RF, K), this is F_RF^H @ H
    FH_F = F_RF.conj().T @ F_RF  # shape (N_RF, N_RF), this is F_RF^H @ F_RF

    gram = FH_H @ FH_H.conj().T + lambda_lag * FH_F  # shape (N_RF, N_RF)

    # Use solve AX=B to find FBB_tilde
    F_BB_tilde = np.linalg.solve(gram, FH_H)  # shape (N_RF, K)
    current_power = np.linalg.norm(F_RF @ F_BB_tilde, ord='fro') ** 2
    beta = np.sqrt(P_t / current_power)

    F_BB = beta * F_BB_tilde
    return F_BB, F_BB_tilde, beta

# Gives Batch wise F_BB, F_BB_tilde and beta for given H, F_RF, noise_power and P_t
def kkt_digital_precoder_torch(
    H: torch.Tensor,       # (batch, M, K) complex
    F_RF: torch.Tensor,    # (batch, M, N_RF) complex
    noise_power: float,
    P_t: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, M, K = H.shape
    N_RF = F_RF.shape[2]
    lambda_lag = K * noise_power / P_t

    FH_H = F_RF.conj().transpose(-2, -1) @ H          # (batch, N_RF, K)
    FH_F = F_RF.conj().transpose(-2, -1) @ F_RF        # (batch, N_RF, N_RF)

    gram = FH_H @ FH_H.conj().transpose(-2, -1) + lambda_lag * FH_F
    ridge = eps * torch.eye(N_RF, dtype=F_RF.dtype, device=F_RF.device).unsqueeze(0)
    gram_reg = gram + ridge

    F_BB_tilde = torch.linalg.solve(gram_reg, FH_H)    # (batch, N_RF, K)

    # Power normalization: beta = sqrt(P_t / ||F_RF F_BB_tilde||_F^2)
    product = F_RF @ F_BB_tilde                        # (batch, M, K)
    power = torch.sum(torch.abs(product) ** 2, dim=(-2, -1))  # (batch,)
    beta = torch.sqrt(P_t / power)                     # (batch,)

    F_BB = beta.view(batch, 1, 1) * F_BB_tilde
    return F_BB, F_BB_tilde, beta              

def optimal_beta(
    H:    np.ndarray,   # (M, K) complex
    F_RF: np.ndarray,   # (M, N_RF) complex
    F_BB: np.ndarray,   # (N_RF, K) complex
    noise_power: float,
) -> float:
    K = H.shape[1]
    A = H.conj().T @ F_RF @ F_BB    # (K, K)
    numerator   = np.linalg.norm(A, 'fro') ** 2 + noise_power * K
    denominator = np.real(np.trace(A))
    if abs(denominator) < 1e-12:
        raise ValueError(
            "tr(H^H F_RF F_BB) is near zero — precoder is orthogonal to channel."
        )
    return float(numerator / denominator)

