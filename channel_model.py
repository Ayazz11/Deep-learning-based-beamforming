
#Near-field / far-field channel model
import numpy as np
import matplotlib.pyplot as plt
def array_response_near_field(theta: float, r: float, M: int, d: float, wavelength: float) -> np.ndarray:
    m_idx = np.arange(M) #antenna element indices (0, 1, ..., M-1)
    #print("m_idx:", m_idx);
    delta_m = (2 * m_idx - M + 1) / 2.0  # antenna index relative to array center
    #print("delta_m:", delta_m);
    # distance from m-th antenna element to the target (Eq. 11's r^(m))
    r_m = np.sqrt(r**2 + (delta_m * d) ** 2 - 2 * r * delta_m * d * np.sin(theta)) 
    #print("r_m:", r_m);
    # phase reference is r itself (the array-center distance), per Eq. (11): exp(-j*2pi/lambda*(r^(m) - r))
    phase = -2 * np.pi / wavelength * (r_m-r)
    b = np.exp(1j * phase) / np.sqrt(M)
    return b

def array_response_far_field(theta: float, M: int, d: float, wavelength: float) -> np.ndarray:
    m_idx = np.arange(M)
    delta_m = (2 * m_idx - M + 1) / 2.0
    # phase = -2*pi/wavelength * (r^(m) - r), with (r^(m) - r) -> -delta_m*d*sin(theta)
    phase = -2 * np.pi / wavelength * (-delta_m * d * np.sin(theta))
    b = np.exp(1j * phase) / np.sqrt(M)
    return b

def generate_channel(
    K: int, # number of users
    M: int, # number of antennas
    L: int, # number of paths
    d: float,
    wavelength: float,
    r_min: float ,
    r_max: float ,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    theta_vec_user = np.zeros(K)
    range_vec_user = np.zeros(K)
    H = np.zeros((M, K), dtype=complex)
    for k in range(K):
        h_k = np.zeros(M, dtype=complex)
        for _ in range(L):
            u = rng.uniform(-1.0, 1.0)  # sin(theta)
            theta = np.arcsin(np.clip(u, -1.0, 1.0))
            r = rng.uniform(r_min, r_max)
            alpha = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)  # CN(0,1)
            # NOTE : the model doesn't impose any physical clustering , i.e a scatterer cluster having correlated angle spread across paths'''
            b = array_response_near_field(theta, r, M, d, wavelength)
            common_phase=np.exp(-1j *2 * np.pi / wavelength * r)
            h_k += alpha * common_phase * b
        theta_vec_user[k] = np.rad2deg(theta)
        range_vec_user[k] = r
        h_k *= np.sqrt(M / L)
        H[:, k] = h_k
    return H, theta_vec_user, range_vec_user

def steering_correlation(theta: float, r: float, M: int, d: float, wavelength: float) -> float:
    a_nf = array_response_near_field(theta, r, M, d, wavelength)
    a_ff = array_response_far_field(theta, M, d, wavelength)
    return np.abs(np.vdot(a_nf, a_ff))

def correlation_sweep(theta: float, r_values: np.ndarray, M: int, d: float, wavelength: float) -> np.ndarray:
    return np.array([steering_correlation(theta, r, M, d, wavelength) for r in r_values])

def rayleigh_distance(M: int, d: float, wavelength: float) -> float:
    D = (M - 1) * d
    return 2 * D**2 / wavelength


if __name__ == "__main__":
    print("channel generation...")

    wavelength = 0.003      # 100 GHz
    d = wavelength / 2
    M = 256
    K = 4
    L = 3
    R = rayleigh_distance(M, d, wavelength)
    print("\nRayleigh Distance =", R, "meters")
    theta_deg = 30
    theta = np.deg2rad(theta_deg)
    r=10
    nf_arr1=array_response_near_field(theta=theta, r=r, M=M, d=d, wavelength=wavelength)
    ff_arr1=array_response_far_field(theta=theta, M=M, d=d, wavelength=wavelength)
    print("nf_arr1:", nf_arr1.shape)
    #print("\n",nf_arr1)
    print("\nff_arr1:", ff_arr1.shape)
    #print("\n",ff_arr1)

    r_values = np.arange(5.0, 100.0 + 1e-9, 0.5)
    rho_values = correlation_sweep(theta, r_values, M, d, wavelength)
    idx_nearest = np.argmin(np.abs(r_values - R))
    r_nearest = r_values[idx_nearest]
    rho_nearest = rho_values[idx_nearest]
    print(f"Nearest sampled r to R: r = {r_nearest:.2f} m, rho(r) = {rho_nearest:.4f}")
    print(f"rho(r=5m)  = {rho_values[0]:.4f}")
    print(f"rho(r=60m) = {rho_values[-1]:.4f}")
    print(f"min rho over sweep = {rho_values.min():.4f} at r = {r_values[np.argmin(rho_values)]:.2f} m")


    rng1 = np.random.default_rng(42)
    H1,_,_ = generate_channel(
        K=K,
        M=M,
        L=L,
        d=d,
        wavelength=wavelength,
        r_min=50,
        r_max=150,
        rng=rng1
    )
    rng2 = np.random.default_rng(100)
    H2,_,_ = generate_channel(
        K=K,
        M=M,
        L=L,
        d=d,
        wavelength=wavelength,
        r_min=50,
        r_max=150,
        rng=rng2
    )
    print("Shape of H1:", H1.shape)
    #print("\nChannel Matrix H1:")
    #print(H1)
    print("\nShape of H2:", H2.shape)
    #print("\nChannel Matrix H2:")
    #print(H2)
