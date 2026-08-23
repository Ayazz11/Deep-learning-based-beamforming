import numpy as np
from kkt_precoder import kkt_digital_precoder


def random_digital_precoder(F_RF, N_RF, K, P_t, rng):
    F_BB = (
        rng.standard_normal((N_RF, K))
        +1j*rng.standard_normal((N_RF, K))
    )
    F_BB *= np.sqrt(
        P_t/np.linalg.norm(F_RF@F_BB,'fro')**2
    )
    return F_BB

def random_cm_analog_precoder(M: int, N_RF: int, rng: np.random.Generator | None = None) -> np.ndarray:

    if rng is None:
        rng = np.random.default_rng()
    random_phases = rng.uniform(0, 2 * np.pi, size=(M, N_RF))
    return np.exp(1j * random_phases) / np.sqrt(M)

def zf_digital_precoder(H: np.ndarray, F_RF: np.ndarray, P_t: float) -> np.ndarray:

    H_eq = H.conj().T @ F_RF  # shape (K, N_RF)
    F_BB_unnorm = np.linalg.pinv(H_eq)  # shape (N_RF, K)

    current_power = np.linalg.norm(F_RF @ F_BB_unnorm, ord='fro') ** 2
    scale = np.sqrt(P_t / current_power)
    return scale * F_BB_unnorm

def MF_Analog_precoder(H, N_RF, M):
    return np.exp(1j * np.angle(H[:, :N_RF])) / np.sqrt(M)

def th_hmp_precoder(H, N_RF, noise_power, P_t, N_iter=8, eps=1e-6):
    M, K = H.shape
    lambda_lag = K * noise_power / P_t

    # Initialize F_RF randomly (CM constraint)
    F_RF = np.exp(1j * np.random.uniform(0, 2*np.pi, (M, N_RF))) / np.sqrt(M)

    for _ in range(N_iter):
        # Step 1: Given F_RF, compute optimal F_BB via KKT (same as our kkt_precoder)
        FH_H = F_RF.conj().T @ H          # (N_RF, K)
        FH_F = F_RF.conj().T @ F_RF        # (N_RF, N_RF)
        gram = FH_H @ FH_H.conj().T + lambda_lag * FH_F
        F_BB_tilde = np.linalg.solve(gram + eps*np.eye(N_RF), FH_H)

        # Step 2: Update F_RF by finding dominant singular vectors of
        # the gradient matrix (trace maximization step)
        # Gradient direction: H @ F_BB_tilde^H
        G = H @ F_BB_tilde.conj().T         # (M, N_RF)
        U, _, Vh = np.linalg.svd(G, full_matrices=False)
        F_RF_new = U[:, :N_RF] @ Vh[:N_RF, :N_RF]  # (M, N_RF) -- unitary

    # Project to CM constraint: keep phase, normalize amplitude
    F_RF = np.exp(1j * np.angle(F_RF_new)) / np.sqrt(M)
    # Final F_BB with power normalization
    F_BB, _, beta = kkt_digital_precoder(H, F_RF, noise_power, P_t)
    return F_RF, F_BB

def build_near_field_dictionary(
    M: int,
    d: float,
    wavelength: float,
    array_response_fn,
    theta_grid: np.ndarray | None = None,
    r_grid: np.ndarray | None = None,
    n_theta: int = 181,
    n_r: int = 200,
    r_min: float = 5.0,
    r_max: float = 80.0,
) -> np.ndarray:
    """Precompute the near-field polar-domain steering-vector dictionary
    (El Ayach et al., IEEE TWC 2014; near-field/polar-domain grid per
    Cui & Dai, IEEE TCOM 2022). Channel-independent -- build ONCE and reuse
    across a Monte Carlo comparison loop rather than rebuilding per-channel.

    array_response_fn: your array_response_near_field(theta, r, M, d, wavelength)
    from channel_model.py, passed in to avoid a circular import here.

    Returns
    -------
    A_dict : (M, n_theta*n_r) complex ndarray. Columns are unit-norm,
             unit-modulus-per-entry (CM-feasible) -- selecting any column
             directly as an F_RF column requires no further projection.
    """
    if theta_grid is None:
        u_grid = np.linspace(-1.0, 1.0, n_theta)
        theta_grid = np.arcsin(u_grid)
    if r_grid is None:
        s = np.arange(1, n_r + 1)
        r_grid = r_min + (r_max - r_min) * (1.0 - 1.0 / s) / (1.0 - 1.0 / n_r)

    atoms = [
        array_response_fn(theta, r, M, d, wavelength)
        for r in r_grid
        for theta in theta_grid
    ]
    return np.stack(atoms, axis=1)  # (M, n_theta*n_r)


def somp_hybrid_precoder(
    H: np.ndarray,
    N_RF: int,
    noise_power: float,
    P_t: float,
    A_dict: np.ndarray,
    kkt_digital_precoder_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """SOMP (Simultaneous Orthogonal Matching Pursuit) hybrid precoder.

    Approximates the ideal fully-digital MSE-minimizing precoder F_opt under
    the CM-constrained hybrid architecture by greedily selecting N_RF columns
    of a fixed steering-vector dictionary A_dict, following El Ayach et al.,
    "Spatially Sparse Precoding in Millimeter Wave MIMO Systems," IEEE Trans.
    Wireless Commun., vol. 13, no. 3, pp. 1499-1513, Mar. 2014 (Algorithm 1).

    Parameters
    ----------
    H : (M, K) complex ndarray -- downlink channel, one column per user.
    N_RF : number of RF chains == number of SOMP iterations.
    noise_power, P_t : as elsewhere in this codebase.
    A_dict : (M, n_atoms) precomputed dictionary from build_near_field_dictionary().
    kkt_digital_precoder_fn : your kkt_digital_precoder(H, F_RF, noise_power, P_t)
        -> (F_BB, F_BB_tilde, beta), passed in to avoid a circular import here.

    Returns
    -------
    F_RF : (M, N_RF) complex ndarray -- CM-feasible, columns are dictionary atoms.
    F_BB : (N_RF, K) complex ndarray -- power-normalized so ||F_RF @ F_BB||_F^2 = P_t.
    """
    M, K = H.shape

    # Step 1: ideal (fully-digital) MSE-minimizing precoder, F_RF = I
    F_opt, _, _ = kkt_digital_precoder_fn(H, np.eye(M, dtype=complex), noise_power, P_t)

    # Step 2: greedy atom selection against the residual
    F_res = F_opt.copy()
    selected_idx: list[int] = []
    F_RF = np.zeros((M, 0), dtype=complex)
    F_BB = np.zeros((0, K), dtype=complex)

    for _ in range(N_RF):
        # correlation of every atom with the residual, summed over all K
        # users -> pick the atom explaining the most residual energy jointly
        Psi = A_dict.conj().T @ F_res                  # (n_atoms, K)
        scores = np.sum(np.abs(Psi) ** 2, axis=1)       # (n_atoms,)
        scores[selected_idx] = -1.0                      # never reselect
        k_star = int(np.argmax(scores))
        selected_idx.append(k_star)

        F_RF = A_dict[:, selected_idx]                   # (M, n_selected)
        F_BB = np.linalg.pinv(F_RF) @ F_opt               # LS digital update
        F_res = F_opt - F_RF @ F_BB

    # Step 3: final power normalization (F_RF fixed/hardware, only F_BB scaled)
    F_BB = np.sqrt(P_t) * F_BB / np.linalg.norm(F_RF @ F_BB, ord="fro")

    return F_RF, F_BB