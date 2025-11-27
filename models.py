"""
Model Definitions and Logistic Map Core Functions
Includes: Logistic map simulation, data generation, various regression models
"""
import numpy as np
from typing import Tuple, List, Optional
from tqdm.auto import tqdm
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = type('nn', (), {'Module': object})


# ============================================================================
# Logistic Map Core Functions
# ============================================================================

def logistic_map_next(x: float, r: float) -> float:
    """Logistic map iteration: x_{n+1} = r * x_n * (1 - x_n)"""
    return r * x * (1.0 - x)


def simulate_logistic_series(r: float, num_iterations: int, 
                            num_discard: int, x0: float) -> np.ndarray:
    """
    Simulate logistic map series
    
    Args:
        r: Parameter r
        num_iterations: Total number of iterations
        num_discard: Number of initial steps to discard
        x0: Initial value
        
    Returns:
        Retained sequence values
    """
    x = x0
    values = []
    for i in range(num_iterations):
        x = logistic_map_next(x, r)
        if i >= num_discard:
            values.append(x)
    return np.asarray(values, dtype=np.float64)


def generate_bifurcation_data(r_values: np.ndarray, num_iterations: int,
                              num_discard: int, seed: int, 
                              progress: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate bifurcation diagram data
    
    Returns:
        (r_points, x_points): r values and x values of all sampled points
    """
    rng = np.random.default_rng(seed)
    all_r_points, all_x_points = [], []
    
    iterator = tqdm(r_values, desc="bifurcation", disable=not progress)
    for r in iterator:
        x0 = rng.uniform(0.1, 0.9)
        xs = simulate_logistic_series(r, num_iterations, num_discard, x0)
        sample_count = min(300, xs.shape[0])
        sampled = xs[-sample_count:]
        all_x_points.append(sampled)
        all_r_points.append(np.full(sampled.shape[0], r))
    
    return np.concatenate(all_r_points), np.concatenate(all_x_points)


def build_density_matrix(r_values: np.ndarray, num_iterations: int,
                        num_discard: int, num_bins: int, seed: int,
                        progress: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build density matrix
    
    Returns:
        (densities, bin_edges): Density matrix and bin edges
    """
    rng = np.random.default_rng(seed)
    densities = np.zeros((r_values.shape[0], num_bins), dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    
    iterator = enumerate(tqdm(r_values, desc="density", disable=not progress))
    for idx, r in iterator:
        x0 = rng.uniform(0.1, 0.9)
        xs = simulate_logistic_series(r, num_iterations, num_discard, x0)
        hist, _ = np.histogram(xs, bins=bin_edges, range=(0.0, 1.0))
        hist = hist.astype(np.float64)
        hist /= max(hist.sum(), 1.0)
        densities[idx] = hist
    
    return densities, bin_edges


def compute_lyapunov_for_r_values(r_values: np.ndarray, num_iterations: int,
                                  num_discard: int, seed: int,
                                  progress: bool) -> np.ndarray:
    """
    Compute Lyapunov exponents for each r value
    
    Returns:
        Array of Lyapunov exponents
    """
    rng = np.random.default_rng(seed)
    lyap = np.zeros_like(r_values, dtype=np.float64)
    
    iterator = enumerate(tqdm(r_values, desc="lyapunov", disable=not progress))
    for idx, r in iterator:
        x0 = rng.uniform(0.1, 0.9)
        xs = simulate_logistic_series(r, num_iterations, num_discard, x0)
        if xs.size == 0:
            lyap[idx] = np.nan
            continue
        # λ = <ln|f'(x)|>
        deriv = np.abs(r * (1.0 - 2.0 * xs))
        deriv = np.clip(deriv, 1e-12, None)
        lyap[idx] = float(np.mean(np.log(deriv)))
    
    return lyap


# ============================================================================
# Traditional Machine Learning Models
# ============================================================================

def train_mlp_regressor(X_train: np.ndarray, y_train: np.ndarray,
                       hidden_layers: Tuple[int, ...], max_iter: int,
                       learning_rate_init: float, seed: int,
                       verbose: bool = False) -> MLPRegressor:
    """Train MLP regressor"""
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(512, max(64, X_train.shape[0] // 20)),
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=seed,
        verbose=verbose,
    )
    model.fit(X_train, y_train)
    return model


def train_regressor(X_train: np.ndarray, y_train: np.ndarray, model_type: str,
                   hidden_layers: Tuple[int, ...], max_iter: int,
                   learning_rate_init: float, seed: int, verbose: bool = False):
    """
    Train regressor (supports multiple types)
    
    Args:
        model_type: 'mlp', 'ridge', 'svr'
    """
    if model_type == "ridge":
        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(X_train, y_train)
        return model
    
    if model_type == "svr":
        # Single-output SVR
        if y_train.ndim == 1 or y_train.shape[1] == 1:
            svr = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.001)
            svr.fit(X_train, y_train.ravel())
            return svr
        
        # Multi-output SVR wrapper
        models = []
        for j in range(y_train.shape[1]):
            svr = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.001)
            svr.fit(X_train, y_train[:, j])
            models.append(svr)
        
        class _MultiSVR:
            def __init__(self, ms):
                self.ms = ms
            def predict(self, X):
                return np.column_stack([m.predict(X) for m in self.ms])
        return _MultiSVR(models)
    
    # Default to MLP
    return train_mlp_regressor(X_train, y_train, hidden_layers, max_iter,
                              learning_rate_init, seed, verbose)


# ============================================================================
# GRU-MLP Hybrid Model (Deep Learning)
# ============================================================================

class GRUMLPModel(nn.Module):
    """
    GRU+MLP hybrid model
    - GRU encodes time series
    - MLP fuses r features and makes predictions
    - Optional physical residual connection
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float, mlp_hidden: Tuple[int, ...], output_size: int,
                 residual: bool = True, r_feat_dim: int = 1,
                 mlp_dropout: float = 0.0, spectral_norm: bool = False):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required to use GRU model")
        
        # GRU encoder
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Build MLP head
        layers = []
        in_dim = hidden_size + r_feat_dim
        for h in mlp_hidden:
            lin = nn.Linear(in_dim, h)
            if spectral_norm:
                try:
                    from torch.nn.utils.parametrizations import spectral_norm as _sn
                    lin = _sn(lin)
                except:
                    pass
            layers.extend([lin, nn.ReLU()])
            if mlp_dropout > 0:
                layers.append(nn.Dropout(p=mlp_dropout))
            in_dim = h
        
        out_lin = nn.Linear(in_dim, output_size)
        if spectral_norm:
            try:
                from torch.nn.utils.parametrizations import spectral_norm as _sn
                out_lin = _sn(out_lin)
            except:
                pass
        layers.append(out_lin)
        
        self.mlp = nn.Sequential(*layers)
        self.horizon = output_size
        self.residual = residual
    
    def forward(self, seq_x: torch.Tensor, r_feat: torch.Tensor,
                r_scalar: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            seq_x: [B, T, D] Sequence input
            r_feat: [B, R] r features
            r_scalar: [B, 1] r scalar (for residual)
        """
        _, h_T = self.gru(seq_x)
        h_last = h_T[-1]  # [B, H]
        x = torch.cat([h_last, r_feat], dim=1)
        mlp_out = self.mlp(x)
        
        if not self.residual:
            return torch.clip(mlp_out, 0.0, 1.0)
        
        # Physical residual: baseline prediction based on logistic map
        last_x = seq_x[:, -1:, 0]
        base_steps = []
        cur = last_x
        for _ in range(self.horizon):
            cur = r_scalar * cur * (1.0 - cur)
            base_steps.append(cur)
        base = torch.cat(base_steps, dim=1)
        
        return torch.clip(base + mlp_out, 0.0, 1.0)


class GRUMLPRegressor:
    """GRU-MLP Regressor Wrapper"""
    
    def __init__(self, window: int, horizon: int, hidden_size: int,
                 mlp_hidden: Tuple[int, ...], num_layers: int = 1,
                 dropout: float = 0.0, lr: float = 1e-3, batch_size: int = 256,
                 max_epochs: int = 300, seed: int = 42, device: Optional[str] = None,
                 residual: bool = True, residual_lambda: float = 1e-3,
                 poly_degree: int = 1, input_jitter: float = 0.0,
                 grad_clip: float = 1.0, r_min: float = 2.5, r_max: float = 4.0,
                 r_poly_degree: int = 1, r_sin_cos: bool = False,
                 r_norm_to_unit: bool = False, weight_decay: float = 0.0,
                 rollout_k: int = 0, rollout_lambda: float = 0.0,
                 mlp_dropout: float = 0.0, spectral_norm: bool = False):
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required: pip install torch")
        
        self.window = window
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.mlp_hidden = mlp_hidden
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.seed = seed
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.residual = residual
        self.residual_lambda = residual_lambda
        self.poly_degree = max(1, int(poly_degree))
        self.input_jitter = float(input_jitter)
        self.grad_clip = float(grad_clip)
        self.r_min, self.r_max = r_min, r_max
        self.r_poly_degree = max(1, int(r_poly_degree))
        self.r_sin_cos = bool(r_sin_cos)
        self.r_norm_to_unit = bool(r_norm_to_unit)
        self.weight_decay = float(weight_decay)
        self.rollout_k = int(rollout_k)
        self.rollout_lambda = float(rollout_lambda)
        
        r_feat_dim = self._r_feat_dim()
        self.model = GRUMLPModel(
            input_size=self.poly_degree,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            mlp_hidden=mlp_hidden,
            output_size=horizon,
            residual=residual,
            r_feat_dim=r_feat_dim,
            mlp_dropout=mlp_dropout,
            spectral_norm=spectral_norm
        ).to(self.device)
        
        self.loss_fn = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)
    
    def _r_feat_dim(self) -> int:
        """Calculate r feature dimension"""
        dim = self.r_poly_degree
        if self.r_sin_cos:
            dim += 2
        return dim
    
    def _build_r_features(self, r_np: np.ndarray) -> np.ndarray:
        """Build polynomial and trigonometric features for r"""
        if self.r_norm_to_unit:
            r_scaled = (2.0 * (r_np - self.r_min) / max(self.r_max - self.r_min, 1e-6)) - 1.0
        else:
            r_scaled = r_np
        
        feats = [r_scaled]
        for p in range(2, self.r_poly_degree + 1):
            feats.append(np.power(r_scaled, p))
        if self.r_sin_cos:
            feats.append(np.sin(np.pi * r_np))
            feats.append(np.cos(np.pi * r_np))
        return np.concatenate(feats, axis=1)
    
    def _to_tensors(self, X: np.ndarray, Y: np.ndarray):
        """Convert numpy arrays to PyTorch tensors"""
        r = X[:, 0:1].astype(np.float32)
        seq = X[:, 1:].astype(np.float32)
        y = Y.astype(np.float32)
        
        # Sequence polynomial features
        if self.poly_degree > 1:
            feats = [seq]
            for p in range(2, self.poly_degree + 1):
                feats.append(np.clip(np.power(seq, p), 0.0, 1.0))
            seq_t = torch.from_numpy(np.stack(feats, axis=-1))
        else:
            seq_t = torch.from_numpy(seq).unsqueeze(-1)
        
        r_scalar_t = torch.from_numpy(r)
        r_feat = self._build_r_features(r)
        r_feat_t = torch.from_numpy(r_feat.astype(np.float32))
        y_t = torch.from_numpy(y)
        
        return seq_t, r_feat_t, r_scalar_t, y_t
    
    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False):
        """
        Train model
        
        Args:
            X: [N, 1+window] Input (r + window sequence)
            Y: [N, horizon] Target
            verbose: Whether to print training information
        """
        if X.shape[0] == 0:
            return
        
        seq_t, r_feat_t, r_scalar_t, y_t = self._to_tensors(X, Y)
        
        # Input jitter augmentation
        if self.input_jitter > 0:
            noise = torch.randn_like(seq_t) * self.input_jitter
            seq_t = torch.clamp(seq_t + noise, 0.0, 1.0)
        
        ds = TensorDataset(seq_t, r_feat_t, r_scalar_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        best_loss = float("inf")
        patience_counter = 0
        patience = 20
        
        for epoch in range(self.max_epochs):
            loss_sum, count = 0.0, 0
            
            for seq_b, r_feat_b, r_scalar_b, y_b in dl:
                seq_b = seq_b.to(self.device)
                r_feat_b = r_feat_b.to(self.device)
                r_scalar_b = r_scalar_b.to(self.device)
                y_b = y_b.to(self.device)
                
                self.opt.zero_grad()
                pred = self.model(seq_b, r_feat_b, r_scalar_b)
                loss = self.loss_fn(pred, y_b)
                
                # Residual regularization
                if self.residual and self.residual_lambda > 0:
                    base = self._compute_base(seq_b, r_scalar_b)
                    res = pred - base
                    loss = loss + self.residual_lambda * torch.mean(res * res)
                
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                   self.grad_clip)
                self.opt.step()
                
                loss_sum += float(loss.detach().cpu()) * seq_b.size(0)
                count += seq_b.size(0)
            
            epoch_loss = loss_sum / max(count, 1)
            
            # Early stopping
            if epoch_loss < best_loss - 1e-8:
                best_loss = epoch_loss
                patience_counter = 0
                best_state = {k: v.detach().cpu() for k, v in 
                            self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{self.max_epochs}, Loss: {epoch_loss:.6e}")
        
        # Restore best model
        if 'best_state' in locals():
            self.model.load_state_dict({k: v.to(self.device) for k, v in 
                                       best_state.items()})
    
    def _compute_base(self, seq_b: torch.Tensor, r_b: torch.Tensor) -> torch.Tensor:
        """Compute physical baseline (logistic map extrapolation)"""
        last_x = seq_b[:, -1:, 0]
        steps = []
        cur = last_x
        for _ in range(self.horizon):
            cur = r_b * cur * (1.0 - cur)
            steps.append(cur)
        return torch.cat(steps, dim=1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict
        
        Args:
            X: [N, 1+window] Input
            
        Returns:
            [N, horizon] Predictions
        """
        if X.shape[0] == 0:
            return np.empty((0, self.horizon), dtype=np.float32)
        
        self.model.eval()
        seq_t, r_feat_t, r_scalar_t, _ = self._to_tensors(
            X, np.zeros((X.shape[0], self.horizon), dtype=np.float32))
        
        preds = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 4096):
                seq_b = seq_t[i:i+4096].to(self.device)
                r_feat_b = r_feat_t[i:i+4096].to(self.device)
                r_scalar_b = r_scalar_t[i:i+4096].to(self.device)
                out = self.model(seq_b, r_feat_b, r_scalar_b)
                preds.append(out.detach().cpu().numpy())
        
        return np.concatenate(preds, axis=0)
