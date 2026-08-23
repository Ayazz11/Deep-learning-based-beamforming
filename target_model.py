
#Near-field / far-field channel model
import numpy as np

def array_response_near_field(theta: float, r: float, M: int, d: float, wavelength: float) -> np.ndarray:
    m_idx = np.arange(M)
    delta_m = (2 * m_idx - M + 1) / 2.0  # antenna index relative to array center
    # distance from m-th antenna element to the target (Eq. 11's r^(m))
    r_m = np.sqrt(r**2 + (delta_m * d) ** 2 - 2 * r * delta_m * d * np.sin(theta)) 
    # phase reference is r itself (the array-center distance), per Eq. (11): exp(-j*2pi/lambda*(r^(m) - r))
    phase = -2 * np.pi / wavelength * (r_m-r)
    b = np.exp(1j * phase) / np.sqrt(M)
    return b

def generate_target_channel(
    T: int, #no. of targets 
    M: int, #no. of BS antennas
    d: float, #antenna spacing
    wavelength: float,
    r_min: float,
    r_max: float ,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    theta_vec_target = np.zeros(T)
    range_vec_target = np.zeros(T)
    B = np.zeros((M, T), dtype=complex)
    for t in range(T):
        u = rng.uniform(-1.0, 1.0)  # sin(theta)
        theta = np.arcsin(np.clip(u, -1.0, 1.0))
        r = rng.uniform(r_min, r_max)
    # NOTE : the model doesn't impose any physical clustering , i.e a scatterer cluster having correlated angle spread across paths'''
        b_t = array_response_near_field(theta, r, M, d, wavelength)
        B[:, t] = b_t
        theta_vec_target[t] = np.rad2deg(theta)
        range_vec_target[t] = r
    return B, theta_vec_target, range_vec_target

if __name__ == "__main__":
    print("Testing channel generation...")

    wavelength = 0.01      # 30 GHz
    d = wavelength / 2
    M = 8
    T = 1
    theta_deg = 30
    theta = np.deg2rad(theta_deg)
    r=100
    nf_arr1=array_response_near_field(theta=theta, r=10, M=M, d=d, wavelength=wavelength)
    #print("nf_arr1:", nf_arr1.shape)
    #print("\n",nf_arr1)
    #B1=generate_target_channel(T=T, M=M, d=d, wavelength=wavelength, r_min=5.0, r_max=80.0)
    #print("B1:", B1.shape)
    #print("\n",B1)
    rng1 = np.random.default_rng(42)

   