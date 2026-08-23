import numpy as np
from channel_model import array_response_near_field
def generate_channel_los(
    K: int,
    M: int,
    d: float,
    wavelength: float,
    r_min: float,
    r_max: float,
    rng: np.random.Generator | None = None,
    r_ref: float = 5.0,
    path_loss_exponent: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if rng is None:
        rng = np.random.default_rng()
    theta_vec_user = np.zeros(K)
    range_vec_user = np.zeros(K)
    H = np.zeros((M, K), dtype=complex)

    for k in range(K):
        u = rng.uniform(-1.0, 1.0)  # sin(theta)
        theta = np.arcsin(np.clip(u, -1.0, 1.0))
        r = rng.uniform(r_min, r_max)
        phi = rng.uniform(0.0, 2 * np.pi)  # unknown overall propagation phase

        path_loss = (r_ref / r) ** path_loss_exponent
        alpha = np.sqrt(M * path_loss) * np.exp(1j * phi)

        b = array_response_near_field(theta, r, M, d, wavelength)
        H[:, k] = alpha * b

        theta_vec_user[k] = np.rad2deg(theta)
        range_vec_user[k] = r

    return H, theta_vec_user, range_vec_user