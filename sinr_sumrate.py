
import numpy as np
import torch
def compute_sinr(H: np.ndarray, F_RF: np.ndarray, F_BB: np.ndarray, noise_power: float) -> np.ndarray:
    K = H.shape[1]
    # Effective channel after analog+digital precoding, for every (k, i) pair:
    # signal_term[k, i] = h_k^H @ F_RF @ f_BB,i
    # H^H @ F_RF @ F_BB has shape (K, K); entry [k, i] = h_k^H F_RF f_BB,s
    effective = H.conj().T @ F_RF @ F_BB  # shape (K, K)
    power_matrix = np.abs(effective) ** 2  # |h_k^H F_RF f_BB,i|^2, shape (K, K)

    signal_power = np.diag(power_matrix)  # i == k terms
    total_power = power_matrix.sum(axis=1)  # sum over all i for each k
    interference_power = total_power - signal_power
    sinr = signal_power / (interference_power + noise_power) 
    return sinr

def sum_rate(H: np.ndarray, F_RF: np.ndarray, F_BB: np.ndarray, noise_power: float) -> float:

    sinr = compute_sinr(H, F_RF, F_BB, noise_power)
    return np.sum(np.log2(1.0 + sinr))

def sum_rate_torch(
    H: torch.Tensor,        # (batch, M, K) complex
    F_RF: torch.Tensor,     # (batch, M, N_RF) complex
    F_BB: torch.Tensor ,    # (batch, Nrf, K) complex
    noise_power: float,
    P_t: float,
) -> torch.Tensor:                                # (batch, N_RF, K)

    # Effective channel
    #
    # Hᴴ : (batch, K, M)
    # F_RF : (batch, M, N_RF)
    # F_BB : (batch, N_RF, K)
    #
    # effective = Hᴴ F_RF F_BB
    #
    effective = H.conj().transpose(1, 2) @ F_RF @ F_BB
    # (batch, K, K)

    # Desired signal power
    signal = torch.abs(torch.diagonal(effective, dim1=1, dim2=2)) ** 2
    # (batch, K)

    # Total received power
    total = torch.abs(effective) ** 2
    # (batch, K, K)

    total_power = total.sum(dim=2)
    # (batch, K)

    # Interference
    interference = total_power - signal

    # SINR
    sinr = signal / (interference + noise_power)

    # Per-user rate
    rates = torch.log2(1 + sinr)
    # (batch, K)

    # Sum-rate for each channel realization
    sum_rate_per_sample = rates.sum(dim=1)
    # (batch,)

    # Mean over batch
    return sum_rate_per_sample.mean()

def sensing_power_torch(
    B: torch.Tensor,
    F_RF: torch.Tensor,
    F_BB: torch.Tensor,
)-> torch.Tensor:
    TX = F_RF @ F_BB        # (batch,M,K)
    projections = (B.conj().transpose(-2,-1)@ TX)                         # (batch,T,K)
    power = torch.sum(
        torch.abs(projections)**2,
        dim=-1
    )       
    target_avg = power.mean(dim=0)        # (T,)                 # (batch,T)
    Batch_avg_power = target_avg.mean()   #scalar
    return Batch_avg_power, target_avg