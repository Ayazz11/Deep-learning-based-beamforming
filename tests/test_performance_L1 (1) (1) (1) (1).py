import numpy as np
import torch
import sys
from pathlib import Path
import time
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from channel_model_los import generate_channel_los
from sinr_sumrate import  sum_rate
from precoders_baseline import random_cm_analog_precoder, zf_digital_precoder, MF_Analog_precoder, random_digital_precoder, th_hmp_precoder
from kkt_precoder import kkt_digital_precoder
from loss import kkt_digital_precoder_torch, general_mse_objective, concentrated_mse_objective
from network import MultiuserAnalogPrecoderNet
from channel_model import array_response_near_field
from precoders_baseline import build_near_field_dictionary, somp_hybrid_precoder
MODEL_PATH = PROJECT_ROOT / "models" / "Best_comm_model_L1.pt"
M, K, N_RF, L, T = 128, 4, 4, 1, 0
fc, c = 100e9, 3e8
wavelength = c / fc
d = wavelength / 2
rng = np.random.default_rng()
P_t = 1.0
noise_power = 0.1  # SNR = 10*log10(P_t/noise_power) ~ 10 dB

rate_zf_random_analog = []
rate_random_digital = []
rate_zf_mf = []
rate_random_mf = []
rate_kkt_mf = []
rate_nn=[]
rate_full_digital = []
rate_th_hmp = []
mse_random_zf=[]
mse_random_random=[]
mse_mf_zf=[]
mse_mf_random=[]
mse_kkt=[]
mse_nn=[]
mse_full_digital=[]
mse_th_hmp=[]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiuserAnalogPrecoderNet(
    M=M,
    K=K,
    T=T,
    N_RF=N_RF,
    mlp_dims=[512, 256, 128]
).to(device)
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval();
A_dict = build_near_field_dictionary(M, d, wavelength, array_response_near_field,
                                      n_theta=500, n_r=200, r_min=5.0, r_max=80.0)

time_taken = []
nn_inference_times_ms = []
somp_inference_times_ms = []
for _ in range(500):
    H,_,_ = generate_channel_los(K=K, M=M, d=d, wavelength=wavelength, r_min=5.0, r_max=80.0, rng=rng)
    #config 1 : Random analog precoder AND ZF- Baseband precoder.
    F_RF_random = random_cm_analog_precoder(M, N_RF, rng=rng)
    F_BB_zf = zf_digital_precoder(H, F_RF_random, P_t)
    rate_zf_random_analog.append(sum_rate(H, F_RF_random, F_BB_zf, noise_power))
    mse_random_zf.append(general_mse_objective(H= H, F_RF= F_RF_random,F_BB=F_BB_zf, noise_power=noise_power, P_t=P_t, beta=None))
    #config 1 : Random analog precoder and Random precoder.
    F_BB_random = random_digital_precoder(F_RF=F_RF_random, N_RF=N_RF, K=K, P_t=P_t, rng=rng)
    rate_random_digital.append(sum_rate(H, F_RF_random, F_BB_random, noise_power))
    mse_random_random.append(general_mse_objective(H= H, F_RF= F_RF_random,F_BB=F_BB_random, noise_power=noise_power, P_t=P_t, beta=None))
    #config 3 : Matched filter analog precoder and ZF digital precoder.
    F_RF_mf = MF_Analog_precoder(H=H, N_RF=N_RF, M=M)  # CM-projected matched filter, one column per user
    F_BB_zf_mf = zf_digital_precoder(H, F_RF_mf, P_t)
    rate_zf_mf.append(sum_rate(H, F_RF_mf, F_BB_zf_mf, noise_power))
    mse_mf_zf.append(general_mse_objective(H= H, F_RF= F_RF_mf,F_BB=F_BB_zf, noise_power=noise_power, P_t=P_t, beta=None))
    #config 4:Matched filter analog precoder and Random digital precoder.
    F_BB_random_mf = random_digital_precoder(F_RF=F_RF_mf, N_RF=N_RF, K=K, P_t=P_t, rng=rng)
    rate_random_mf.append(sum_rate(H, F_RF_mf, F_BB_random_mf, noise_power))
    mse_mf_random.append(general_mse_objective(H= H, F_RF= F_RF_mf,F_BB=F_BB_random_mf, noise_power=noise_power, P_t=P_t, beta=None))
    #config 5:Matched filter analog precoder and KKT digital precoder.
    F_BB_kkt, F_BB_tilde, beta= kkt_digital_precoder(H=H, F_RF=F_RF_mf, noise_power=noise_power, P_t=P_t)
    rate_kkt_mf.append(sum_rate(H, F_RF_mf, F_BB_kkt, noise_power))
    mse_kkt.append(general_mse_objective(H= H, F_RF= F_RF_mf,F_BB=F_BB_kkt, noise_power=noise_power, P_t=P_t, beta=None))
    #config 6: Full digital precoder (ZF)
    F_RF_full_digital = np.eye(M, dtype=complex)
    F_BB_full_digital,_,_ = kkt_digital_precoder(H=H, F_RF=F_RF_full_digital, noise_power=noise_power, P_t=P_t)
    rate_full_digital.append(sum_rate(H, F_RF_full_digital, F_BB_full_digital, noise_power))
    mse_full_digital.append(general_mse_objective(H= H, F_RF= F_RF_full_digital,F_BB=F_BB_full_digital, noise_power=noise_power, P_t=P_t, beta=None))
    #config 8: SOMP
    somp_start = time.perf_counter()
    F_RF_somp, F_BB_somp = somp_hybrid_precoder(H, N_RF, noise_power, P_t, A_dict, kkt_digital_precoder)
    somp_inference_times_ms.append((time.perf_counter() - somp_start) * 1000.0)
    rate_th_hmp.append(sum_rate(H, F_RF_somp, F_BB_somp, noise_power))
    mse_th_hmp.append(general_mse_objective(H= H, F_RF= F_RF_somp,F_BB=F_BB_somp, noise_power=noise_power, P_t=P_t, beta=None))

    # Config 7 : Neural-network analog precoder + KKT digital precoder
    H_torch = torch.as_tensor(
        H[np.newaxis],           # (1, M, K)
        dtype=torch.cfloat,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    nn_start = time.perf_counter()
    with torch.no_grad():
            F_RF_nn = model(H_torch)  # (1, M, N_RF)
            F_BB_nn, _, _ = kkt_digital_precoder_torch(H_torch, F_RF_nn, noise_power, P_t)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    nn_inference_times_ms.append((time.perf_counter() - nn_start) * 1000.0)
    # Convert back to NumPy
    F_RF_nn = F_RF_nn.squeeze(0).cpu().numpy()
    F_BB_nn = F_BB_nn.squeeze(0).cpu().numpy()

    # Performance
    rate_nn.append(
        sum_rate(H, F_RF_nn, F_BB_nn, noise_power)
    )
    mse_nn.append(
        general_mse_objective(
            H=H,
            F_RF=F_RF_nn,
            F_BB=F_BB_nn,
            noise_power=noise_power,
            P_t=P_t,
            beta=None
        )
    )

print("\n" + "="*125)
print(f"{'Configuration':<45} {'Avg Sum Rate (bps/Hz)':>20} {'Avg MSE':>15}")
print("="*100)
print(f"{'1. Random F_RF + ZF F_BB':<45} {np.mean(rate_zf_random_analog):>20.4f} {np.mean(mse_random_zf):>15.6f} ")
print(f"{'2. Random F_RF + Random F_BB':<45} {np.mean(rate_random_digital):>20.4f} {np.mean(mse_random_random):>15.6f} ")
print(f"{'3. Matched Filter F_RF + ZF F_BB':<45} {np.mean(rate_zf_mf):>20.4f} {np.mean(mse_mf_zf):>15.6f} {'--':>15} ")
print(f"{'4. Matched Filter F_RF + Random F_BB':<45} {np.mean(rate_random_mf):>20.4f} {np.mean(mse_mf_random):>15.6f} ")
print(f"{'5. Matched Filter F_RF + KKT F_BB':<45} {np.mean(rate_kkt_mf):>20.4f} {np.mean(mse_kkt):>15.6f} ")

print(f"{'6. Neural Network F_RF + KKT F_BB':<45} "
      f"{np.mean(rate_nn):>20.4f} "
      f"{np.mean(mse_nn):>15.6f} ")

print(f"{'7. Full Digital F_RF + ZF F_BB':<45} {np.mean(rate_full_digital):>20.4f} {np.mean(mse_full_digital):>15.6f} ")

print(f"{'8. SOMP F_RF + TH-HMP F_BB':<45} "
      f"{np.mean(rate_th_hmp):>20.4f} "
      f"{np.mean(mse_th_hmp):>15.6f} ")

print("="*100)
print(f"Average NN inference time (F_RF_nn + F_BB_nn): {np.mean(nn_inference_times_ms):.4f} ms/sample")
print(f"Average SOMP inference time (F_RF_somp + F_BB_somp): {np.mean(somp_inference_times_ms):.4f} ms/sample")

results_sum = {
    "Random F_RF + ZF F_BB": np.mean(rate_zf_random_analog),
    "Random F_RF + Random F_BB": np.mean(rate_random_digital),
    "Matched Filter F_RF + ZF F_BB": np.mean(rate_zf_mf),
    "Matched Filter F_RF + Random F_BB": np.mean(rate_random_mf),
    "Matched Filter F_RF + KKT F_BB": np.mean(rate_kkt_mf),
    "Nueral network F_RF + KKT F_BB": np.mean(rate_nn),
    "Full Digital F_RF + ZF F_BB": np.mean(rate_full_digital),
    "SOMP F_RF + TH-HMP F_BB": np.mean(rate_th_hmp),
}

results_mse = {
    "Random F_RF + ZF F_BB": np.mean(mse_random_zf),
    "Random F_RF + Random F_BB": np.mean(mse_random_random),
    "Matched Filter F_RF + ZF F_BB": np.mean(mse_mf_zf),
    "Matched Filter F_RF + Random F_BB": np.mean(mse_mf_random),
    "Matched Filter F_RF + KKT F_BB": np.mean(mse_kkt),
    "Neural Network F_RF + KKT F_BB": np.mean(mse_nn),
    "Full Digital F_RF + ZF F_BB": np.mean(mse_full_digital),
    "SOMP F_RF + SOMP F_BB": np.mean(mse_th_hmp),
}

best_sum_rate = max(results_sum, key=results_sum.get)
print("\nHighest Sum rate:")
print(f"  {best_sum_rate} --> {results_sum[best_sum_rate]:.4f} bps/Hz")

best_mse = min(results_mse, key=results_mse.get)
print("\nLowest MSE:")
print(f"  {best_mse} --> {results_mse[best_mse]:.4f} bps/Hz")

# Testing convergence of general MSE to concentrated Fbb_kkt embedded MSE
general_mse_value=general_mse_objective(H= H, F_RF= F_RF_mf,F_BB=F_BB_kkt, noise_power=noise_power, P_t=P_t, beta=None)
contrained_mse_value=concentrated_mse_objective(H= H, F_RF= F_RF_mf, noise_power=noise_power, P_t=P_t)
print("\n General: ",general_mse_value," Constrained: ",contrained_mse_value, " last iteration result: ", mse_kkt[4])