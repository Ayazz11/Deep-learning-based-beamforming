import torch
import torch.nn as nn
import numpy as np
class ComplexLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super().__init__() #to initialize nn.module which is the parent class
        self.re = nn.Linear(in_features, out_features, bias=False) # Wr
        self.im = nn.Linear(in_features, out_features, bias=False) # Wi
        self.bias = nn.Parameter(
        torch.zeros(out_features, dtype=torch.cfloat)
        )
        self._init_weights() #this is a function to initialize weights for the current layer, function is inherited from parent class, nn.module

    def _init_weights(self):
        # Xavier init scaled for complex (factor of 1/sqrt(2) per component)
        nn.init.xavier_uniform_(self.re.weight)
        nn.init.xavier_uniform_(self.im.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_features) complex -> (batch, out_features) complex""" 
        xr, xi = x.real, x.imag
        y = torch.complex(
        self.re(xr) - self.im(xi),
        self.re(xi) + self.im(xr),
        )
        return y + self.bias


class ComplexBatchNorm(nn.Module):

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        #num_features = the no. of neurons output in the layer , ex: 512. Each output is treated as 2-D vector (Re(), Im())
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable scale: 2x2 matrix per feature (stored as 4 scalars: Vrr, Vri, Vir, Vii)
        self.gamma_rr = nn.Parameter(torch.ones(num_features) / np.sqrt(2))
        self.gamma_ri = nn.Parameter(torch.zeros(num_features))
        self.gamma_ir = nn.Parameter(torch.zeros(num_features))
        self.gamma_ii = nn.Parameter(torch.ones(num_features) / np.sqrt(2))

        # Learnable shift (complex bias)
        self.beta_r = nn.Parameter(torch.zeros(num_features))
        self.beta_i = nn.Parameter(torch.zeros(num_features))

        # Running statistics for inference
        self.register_buffer('running_mean_r', torch.zeros(num_features)) #an ordinary tensor is not moved to device , hence buffer is used for running mean,variance,covariance
        self.register_buffer('running_mean_i', torch.zeros(num_features)) #"This tensor belongs to the model, but it is not trainable."
        self.register_buffer('running_Vrr', torch.ones(num_features))
        self.register_buffer('running_Vii', torch.ones(num_features))
        self.register_buffer('running_Vri', torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, num_features) complex -> (batch, num_features) complex"""
        xr, xi = x.real, x.imag  # (batch, F)

        if self.training:
            mean_r = xr.mean(dim=0)  # (F,)
            mean_i = xi.mean(dim=0)
            xr_c = xr - mean_r
            xi_c = xi - mean_i

            # 2x2 covariance per feature
            Vrr = (xr_c ** 2).mean(dim=0) + self.eps
            Vii = (xi_c ** 2).mean(dim=0) + self.eps
            Vri = (xr_c * xi_c).mean(dim=0)

            # Update running stats
            self.running_mean_r = self.running_mean_r.mul_(1-self.momentum).add_(self.momentum*mean_r.detach())
            self.running_mean_i = self.running_mean_i.mul_(1-self.momentum).add_(self.momentum*mean_i.detach())
            self.running_Vrr = self.running_Vrr.mul_(1-self.momentum).add_(self.momentum*Vrr.detach())
            self.running_Vii = self.running_Vii.mul_(1-self.momentum).add_(self.momentum*Vii.detach())
            self.running_Vri = self.running_Vri.mul_(1-self.momentum).add_(self.momentum*Vri.detach())
        else:
            mean_r = self.running_mean_r
            mean_i = self.running_mean_i
            xr_c = xr - mean_r
            xi_c = xi - mean_i
            Vrr, Vii, Vri = self.running_Vrr, self.running_Vii, self.running_Vri

        # Whitening via closed-form 2x2 matrix square root inverse
        # For 2x2 [[a,b],[b,d]], the square root involves:
        #   tau = sqrt(Vrr*Vii - Vri^2),  s = sqrt(Vrr + Vii + 2*tau)
        #   C^{-1/2} = (1/s) * [[Vii+tau, -Vri], [-Vri, Vrr+tau]] / tau
        tau = torch.sqrt(torch.clamp(Vrr * Vii - Vri ** 2, min=self.eps))
        s = torch.sqrt(Vrr + Vii + 2 * tau)
        denom = tau * s + self.eps

        # Whitened components
        xr_w = ((Vii + tau) * xr_c - Vri * xi_c) / denom
        xi_w = (-Vri * xr_c + (Vrr + tau) * xi_c) / denom

        # Affine transform: gamma (2x2) @ whitened + beta
        xr_out = self.gamma_rr * xr_w + self.gamma_ri * xi_w + self.beta_r
        xi_out = self.gamma_ir * xr_w + self.gamma_ii * xi_w + self.beta_i

        return torch.complex(xr_out, xi_out)


class ComplexTanh(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.tanh(x.real), torch.tanh(x.imag))


class ComplexMLP(nn.Module):

    def __init__(self, in_dim: int, dims: list[int]):
        #in_dim: dimension of previous layer output, initially it will be the input 
        #dims: dimension of hidden layers we want ex: [256,128,64]
        super().__init__()
        blocks = [] # it will contain the layers
        prev = in_dim #remembers how many inputs soming from previous layer
        for d in dims:
            blocks.append(nn.Sequential(
                ComplexLinear(prev, d), #ex: (64,256) ---> (256,128) ---> (128,64)
                ComplexBatchNorm(d),
                ComplexTanh(),
            ))
            prev = d
        self.blocks = nn.ModuleList(blocks) #now blocks becomes a data member of this self class, can be accessed as sel.blocks, contains internally three block
        self.out_dim = dims[-1]  #gives the output dimension as last element of dims[]
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, in_dim) -> (batch, out_dim) """

        for block in self.blocks:
            x = block(x)
        return x


class CMNormalization(nn.Module):
    def __init__(self, M: int, eps: float = 1e-8):
        super().__init__()
        self.M = M
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., M) complex -> (..., M) complex, |entry| = 1/sqrt(M)"""
        return x / (torch.abs(x) + self.eps) / (self.M ** 0.5)
    
class MultiuserAnalogPrecoderNet(nn.Module):
    def __init__(
        self,
        M: int,
        K: int,
        T: int,
        N_RF: int,
        mlp_dims: list[int] = [512, 256, 128],
        eps_cm: float = 1e-8,
        noise_power: float = 0,
    ):
        super().__init__()
        self.M = M
        self.K = K
        self.N_RF = N_RF
        self.T=T
        self.noise_power = noise_power
        self.per_user_mlp = ComplexMLP(in_dim=M, dims=mlp_dims)
        D_P = mlp_dims[-1]
        self.merge_linear = ComplexLinear(in_features=(K + T) * D_P,
                                          out_features=M * N_RF)

        self.cm_norm = CMNormalization(M=M, eps=eps_cm)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[MultiuserAnalogPrecoderNet] M={M}, K={K}, N_RF={N_RF}, T={T}"
              f"mlp_dims={mlp_dims}, params={n_params:,}")

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        batch = H.shape[0]
        assert H.shape[2] == self.K + self.T, \
        f"Expected {self.K+self.T} columns (K={self.K} users + T={self.T} targets), got {H.shape[2]}"
    # H: (batch, M, K+T) -> (batch, K+T, M) -> (batch*(K+T), M)
        H_flat = H.permute(0, 2, 1).reshape(batch * (self.K + self.T), self.M)
    # One call through per_user_mlp for all users, all batch elements
        feat_flat = self.per_user_mlp(H_flat)        # (batch*(K+T), D_P)

        D_P = feat_flat.shape[-1]
        z = feat_flat.view(batch, (self.K + self.T) * D_P)       # (batch, (K+T)*D_P), same layout as the old torch.cat

        F_RF_vec = self.merge_linear(z)            # (batch, M*N_RF)
        F_RF_raw = F_RF_vec.view(batch, self.M, self.N_RF)
        F_RF = self.cm_norm(F_RF_raw)

        return F_RF

def channel_batch_to_tensor(H_batch: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(H_batch, dtype=torch.cfloat, device=device)
