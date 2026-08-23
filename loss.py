import torch
import numpy as np
from kkt_precoder import kkt_digital_precoder_torch, optimal_beta
def general_mse_objective(
    H:           np.ndarray,   # (M, K) complex
    F_RF:        np.ndarray,   # (M, N_RF) complex
    F_BB:        np.ndarray,   # (N_RF, K) complex
    noise_power: float,
    P_t:         float,        # only used for power reporting, not for beta
    beta:        float | None = None,  # if None, derived via optimal_beta()
) -> dict:

    K = H.shape[1]
    if beta is None:
        beta = optimal_beta(H, F_RF, F_BB, noise_power)

    b1 = 1.0 / beta
    b2 = 1.0 / beta ** 2

    A = H.conj().T @ F_RF @ F_BB    # (K, K)  = H^H F_RF F_BB

    term1 = float(K)
    term2 = float(np.real(-b1 * np.conj(np.trace(A))))
    term3 = float(np.real(-b1 * np.trace(A)))
    term4 = float(b2 * np.linalg.norm(A, 'fro') ** 2)
    term5 = float(b2 * noise_power * K)
    E     = term1 + term2 + term3 + term4 + term5
    return E

def concentrated_mse_objective(
    H: np.ndarray, F_RF: np.ndarray, noise_power: float, P_t: float
) -> float:

    K = H.shape[1]
    FH_F = F_RF.conj().T @ F_RF  # (N_RF, N_RF)
    FH_H = F_RF.conj().T @ H  # (N_RF, K)

    inner = FH_H.conj().T @ np.linalg.solve(FH_F, FH_H)  # (K, K)

    coeff = P_t / (K * noise_power)
    matrix = np.eye(K) + coeff * inner
    return np.real(np.trace(np.linalg.inv(matrix)))
# average concentrated MSE over all 1024 channel realizations in the batch.
def concentrated_mse_loss(
    H: torch.Tensor,       # (batch, M, K) complex
    F_RF: torch.Tensor,    # (batch, M, N_RF) complex, CM-normalized
    noise_power: float,
    P_t: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch, M, K = H.shape
    N_RF = F_RF.shape[2]

    # F_RF^H H: (batch, N_RF, K)
    FH_H = F_RF.conj().transpose(-2, -1) @ H

    # F_RF^H F_RF: (batch, N_RF, N_RF) -- the Gram matrix
    FH_F = F_RF.conj().transpose(-2, -1) @ F_RF

    # Ridge regularization
    ridge = eps * torch.eye(N_RF, dtype=F_RF.dtype, device=F_RF.device).unsqueeze(0)
    FH_F_reg = FH_F + ridge

    inner = FH_H.conj().transpose(-2, -1) @ torch.linalg.solve(FH_F_reg, FH_H)

    # coeff = P_t / (K * sigma_n^2)
    coeff = P_t / (K * noise_power)

    # matrix = I_K + coeff * inner: (batch, K, K)
    I_K = torch.eye(K, dtype=H.dtype, device=H.device).unsqueeze(0)
    matrix = I_K + coeff * inner

    # E(F_RF) = tr( matrix^{-1} ): (batch,) real
    matrix_inv = torch.linalg.inv(matrix)
    E = torch.real(torch.diagonal(matrix_inv, dim1=-2, dim2=-1).sum(dim=-1))

    return E.mean()

def isac_loss(H, F_RF, B, noise_power, P_t, lam, K, T):

    L_com = concentrated_mse_loss(H, F_RF, noise_power, P_t)

    with torch.no_grad():
        F_BB_detached, _, _ = kkt_digital_precoder_torch(
            H, F_RF, noise_power, P_t
        )
    TX = F_RF @ F_BB_detached              # (batch, M, K)  -- F_RF has grad, F_BB does not
    B_H = B.conj().transpose(-2, -1)       # (batch, L, M)
    projections = B_H @ TX                 # (batch, L, K)  -- grad flows through F_RF -> TX
    power_per_target = torch.sum(
        torch.abs(projections)**2, dim=-1
    )                                      # (batch, L)
    L_sen = -power_per_target.mean()       # scalar, negative (we minimize this = maximize gain)

    L_com_norm = L_com / K
    L_sen_norm = L_sen / (P_t)            # flip sign and normalize -> [0, 1]

    total = lam * L_com_norm + (1.0 - lam) * L_sen_norm
    return total, L_com.item(), L_sen.item()