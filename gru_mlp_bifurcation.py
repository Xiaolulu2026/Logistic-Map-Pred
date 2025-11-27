import argparse
import math
from dataclasses import dataclass, replace
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from tqdm.auto import tqdm

# 可选的 PyTorch（用于 GRU+MLP 组合）
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # 允许在未安装 torch 的环境中仍可运行非 GRU 模式
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


# 在未安装 torch 时，提供一个最小的占位 nn.Module，避免类定义阶段报错
if nn is None:
    class _DummyModule:
        pass
    class _DummyNN:
        Module = _DummyModule
    nn = _DummyNN()


@dataclass
class ExperimentConfig:
    r_min: float = 2.5
    r_max: float = 4.0
    num_r: int = 400
    split_r: float = 3.56994  # 稀疏区/密集区分界
    num_iterations: int = 3000
    num_discard: int = 1000
    num_bins: int = 200
    seed: int = 42
    hidden_layers: Tuple[int, ...] = (256, 256)
    max_train_iter: int = 1200
    learning_rate_init: float = 1e-3
    use_pca: bool = False
    pca_components: int = 16
    pca_fit_on: str = "sparse"  # 'sparse' | 'all'
    model_type: str = "mlp"  # 'mlp' | 'ridge' | 'gru'
    symmetrize: bool = False
    smooth_sigma: float = 0.0
    scatter_per_r: int = 200
    mode: str = "map"  # 'map' | 'density' | 'ts'
    map_pairs_per_r: int = 1200
    progress: bool = True
    mlp_verbose: bool = False
    # 时间序列预测配置
    ts_window: int = 16
    ts_horizon: int = 16
    ts_strategy: str = "iter"  # 'direct' | 'iter'
    ts_max_windows_per_r: int = 800
    # Lyapunov 指数
    lyapunov: bool = True
    # GRU 相关
    gru_hidden_size: int = 256
    gru_num_layers: int = 1
    gru_dropout: float = 0.0
    gru_batch_size: int = 256
    # 训练区选择：'sparse' 使用 r<=split_r，'dense' 使用 r>split_r
    train_side: str = "sparse"
    # 物理归纳偏置（GRU 残差）
    gru_residual: bool = True
    gru_residual_lambda: float = 1e-3
    # 多项式序列特征与输入扰动
    gru_poly_degree: int = 1  # 1 表示仅 x；2 表示 [x, x^2] ...
    ts_input_jitter: float = 0.0  # 训练时对窗口加性高斯噪声标准差
    # 训练稳定性
    grad_clip: float = 1.0
    # r 特征展开与归一化
    gru_r_poly_degree: int = 2
    gru_r_sin_cos: bool = False
    r_norm_to_unit: bool = False  # True: 线性映射到[-1,1]
    # 优化器权重衰减
    gru_weight_decay: float = 0.0
    # 物理滚动一致性损失
    rollout_consistency_k: int = 4
    rollout_consistency_lambda: float = 1e-3
    # 物理稀疏区合成数据增强
    ts_phys_aug_sparse: bool = False
    ts_phys_aug_rmin: float = 3.45
    ts_phys_aug_rmax: float = 3.6
    ts_phys_aug_per_r: int = 200
    ts_phys_aug_mode: str = "auto"  # auto | manual
    ts_phys_band: float = 0.15  # 增强带宽，围绕 split_r
    # MLP 正则化
    mlp_spectral_norm: bool = False
    mlp_dropout: float = 0.0
    # 日志
    log_file: Optional[str] = None
    log_level: str = "INFO"
    # 垃圾神经元剪枝（参考 doi:10.7498/aps.71.20211626）
    prune_after_train: bool = False
    prune_threshold: float = 0.09
    prune_max_neurons: int = 4
    prune_retrain_epochs: int = 60
    # 课程学习
    use_curriculum: bool = True
    curriculum_stages: int = 3
    # 自动分界检测（基于 Lyapunov）
    auto_split: bool = False
    auto_split_run: int = 5
    auto_split_eps: float = 1e-4
    auto_split_smooth: int = 5
    # 逐r分桶/导出
    per_r_csv: bool = False
    per_r_bucket_bins: int = 4
    # ESN baseline
    esn_reservoir: int = 500
    esn_spectral_radius: float = 0.9
    esn_leak: float = 1.0
    esn_alpha: float = 1e-4
    # Conv1D baseline
    conv_channels: int = 64
    conv_kernel: int = 3
    conv_dropout: float = 0.1
    # 自动双向
    run_both: bool = False


@dataclass
class BaseArtifacts:
    r_values: np.ndarray
    densities_true: np.ndarray
    bin_edges: np.ndarray


def setup_logging(log_file: Optional[str], level: str) -> logging.Logger:
    logger = logging.getLogger("bifurcation")
    if logger.handlers:
        logger.setLevel(level.upper())
        return logger
    logger.setLevel(level.upper())
    fmt = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def logistic_map_next(x: float, r: float) -> float:
    return r * x * (1.0 - x)


def simulate_logistic_series(r: float, num_iterations: int, num_discard: int, x0: float) -> np.ndarray:
    x = x0
    values = []
    for i in range(num_iterations):
        x = logistic_map_next(x, r)
        if i >= num_discard:
            values.append(x)
    return np.asarray(values, dtype=np.float64)


def generate_bifurcation_data(
    r_values: np.ndarray,
    num_iterations: int,
    num_discard: int,
    seed: int,
    progress: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    all_r_points: List[float] = []
    all_x_points: List[float] = []

    iterator = tqdm(r_values, desc="bifurcation true (r)", disable=not progress)
    for r in iterator:
        x0 = rng.uniform(0.1, 0.9)
        xs = simulate_logistic_series(r, num_iterations=num_iterations, num_discard=num_discard, x0=x0)
        sample_count = min(300, xs.shape[0])
        sampled = xs[-sample_count:]
        all_x_points.append(sampled)
        all_r_points.append(np.full(sampled.shape[0], r))

    r_points = np.concatenate(all_r_points, axis=0)
    x_points = np.concatenate(all_x_points, axis=0)
    return r_points, x_points


def build_density_matrix(
    r_values: np.ndarray,
    num_iterations: int,
    num_discard: int,
    num_bins: int,
    seed: int,
    progress: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    densities = np.zeros((r_values.shape[0], num_bins), dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    iterator = enumerate(tqdm(r_values, desc="true density (r)", disable=not progress))
    for idx, r in iterator:
        x0 = rng.uniform(0.1, 0.9)
        xs = simulate_logistic_series(r, num_iterations=num_iterations, num_discard=num_discard, x0=x0)
        hist, _ = np.histogram(xs, bins=bin_edges, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float64)
        hist /= max(hist.sum(), 1.0)
        densities[idx] = hist

    return densities, bin_edges


def symmetrize_densities(densities: np.ndarray) -> np.ndarray:
    return 0.5 * (densities + densities[:, ::-1])


def gaussian_kernel1d(sigma: float, truncate: float = 3.0) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    radius = int(truncate * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    s = kernel.sum()
    if s <= 0:
        return np.array([1.0], dtype=np.float64)
    return kernel / s


def smooth_along_r(densities: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return densities
    kernel = gaussian_kernel1d(sigma)
    pad = kernel.shape[0] // 2
    padded = np.pad(densities, ((pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(densities)
    for b in range(densities.shape[1]):
        out[:, b] = np.convolve(padded[:, b], kernel, mode="valid")
    return out


def normalize_rows_nonnegative(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.maximum(arr, 0.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums < eps] = 1.0
    return arr / row_sums


def sample_scatter_from_densities(
    r_values: np.ndarray,
    bin_edges: np.ndarray,
    densities: np.ndarray,
    samples_per_r: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if r_values.size == 0:
        return np.array([]), np.array([])
    rng = np.random.default_rng(seed)
    lefts = bin_edges[:-1]
    rights = bin_edges[1:]
    r_list: List[np.ndarray] = []
    x_list: List[np.ndarray] = []
    for i, r in enumerate(r_values):
        probs = np.maximum(densities[i], 0.0)
        s = probs.sum()
        if s <= 0:
            continue
        probs = probs / s
        bins = rng.choice(len(probs), size=samples_per_r, p=probs)
        u = rng.random(samples_per_r)
        xs = lefts[bins] + u * (rights[bins] - lefts[bins])
        r_list.append(np.full(samples_per_r, r))
        x_list.append(xs)
    if not r_list:
        return np.array([]), np.array([])
    return np.concatenate(r_list), np.concatenate(x_list)


def build_map_training_data(
    r_values: np.ndarray,
    num_iterations: int,
    num_discard: int,
    seed: int,
    max_pairs_per_r: int,
    progress: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    iterator = tqdm(r_values, desc="train map data (r)", disable=not progress)
    for r in iterator:
        x0 = rng.uniform(0.1, 0.9)
        xs = simulate_logistic_series(r, num_iterations=num_iterations, num_discard=num_discard, x0=x0)
        if xs.shape[0] < 2:
            continue
        t = min(max_pairs_per_r, xs.shape[0] - 1)
        x_t = xs[:t]
        x_next = xs[1:t + 1]
        X_block = np.column_stack([np.full(t, r, dtype=np.float64), x_t])
        X_blocks.append(X_block)
        y_blocks.append(x_next)
    if not X_blocks:
        return np.empty((0, 2)), np.empty((0,))
    X = np.vstack(X_blocks)
    y = np.concatenate(y_blocks)
    return X, y


def train_map_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hidden_layers: Tuple[int, ...],
    max_iter: int,
    learning_rate_init: float,
    seed: int,
    verbose: bool,
) -> MLPRegressor:
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


def simulate_with_surrogate_vectorized(
    model: MLPRegressor,
    r_values: np.ndarray,
    num_iterations: int,
    num_discard: int,
    seed: int,
    progress: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_r = r_values.shape[0]
    T_keep = max(0, num_iterations - num_discard)
    if T_keep == 0 or num_r == 0:
        return np.zeros((num_r, 0), dtype=np.float64)
    x = rng.uniform(0.1, 0.9, size=num_r)
    kept = np.zeros((num_r, T_keep), dtype=np.float64)
    keep_idx = 0
    iterator = tqdm(range(num_iterations), desc="simulate via fθ", disable=not progress)
    for i in iterator:
        inputs = np.column_stack([r_values, x])
        x_next = model.predict(inputs)
        x_next = np.clip(x_next, 0.0, 1.0)
        x = x_next
        if i >= num_discard:
            kept[:, keep_idx] = x
            keep_idx += 1
    return kept


def train_mlp_on_sparse_region(
    r_values: np.ndarray,
    densities: np.ndarray,
    split_r: float,
    hidden_layers: Tuple[int, ...],
    max_iter: int,
    learning_rate_init: float,
    seed: int,
) -> Tuple[MLPRegressor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = r_values.reshape(-1, 1)
    y = densities
    train_mask = r_values <= split_r
    test_mask = r_values > split_r
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(64, max(8, X_train.shape[0] // 10)),
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=seed,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model, X_train, y_train, X_test, y_test


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    hidden_layers: Tuple[int, ...],
    max_iter: int,
    learning_rate_init: float,
    seed: int,
    verbose: bool,
):
    if model_type == "ridge" or model_type == "ar":
        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(X_train, y_train)
        return model
    if model_type == "svr":
        if y_train.ndim == 1 or y_train.shape[1] == 1:
            svr = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.001)
            svr.fit(X_train, y_train.ravel())
            return svr
        models = []
        for j in range(y_train.shape[1]):
            svr = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.001)
            svr.fit(X_train, y_train[:, j])
            models.append(svr)
        class _MultiSVR:
            def __init__(self, ms): self.ms = ms
            def predict(self, X): return np.column_stack([m.predict(X) for m in self.ms])
        return _MultiSVR(models)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(64, max(8, X_train.shape[0] // 10)),
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=seed,
        verbose=verbose,
    )
    model.fit(X_train, y_train)
    return model


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return 0.5 * (kl_pm + kl_qm)

def ssim_global(img1: np.ndarray, img2: np.ndarray, c1: float = 1e-4, c2: float = 9e-4) -> float:
    # 简化版 SSIM（全图），img 需为浮点，范围[0,1]
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    # 归一化到[0,1]
    x = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    mu_x = x.mean()
    mu_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov_xy = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if den == 0:
        return float("nan")
    return float(num / den)

def emd_1d_rows(hist_true: np.ndarray, hist_pred: np.ndarray, bin_edges: np.ndarray) -> float:
    # 1D EMD：行级累计差的 L1 积分平均
    lefts = bin_edges[:-1]
    rights = bin_edges[1:]
    bin_widths = rights - lefts
    emds = []
    for i in range(hist_true.shape[0]):
        p = np.maximum(hist_true[i], 0.0)
        q = np.maximum(hist_pred[i], 0.0)
        if p.sum() <= 0 and q.sum() <= 0:
            continue
        p = p / max(p.sum(), 1e-12)
        q = q / max(q.sum(), 1e-12)
        cdf_p = np.cumsum(p)
        cdf_q = np.cumsum(q)
        emd = float(np.sum(np.abs(cdf_p - cdf_q) * bin_widths))
        emds.append(emd)
    return float(np.mean(emds)) if emds else float("nan")

def detect_period_from_traj(xs: np.ndarray, max_p: int = 8, tol: float = 0.02) -> int:
    # 返回估计主周期，若不满足阈值返回0
    T = xs.shape[0]
    if T < 2:
        return 0
    best_p = 0
    best_err = 1e9
    for p in range(1, max_p + 1):
        diffs = np.abs(xs[p:] - xs[:-p])
        if diffs.size == 0:
            continue
        err = float(diffs.mean())
        if err < best_err:
            best_err = err
            best_p = p
    return best_p if best_err <= tol else 0

def period_consistency_f1(true_trajs: np.ndarray, pred_trajs: np.ndarray, max_p: int, tol: float) -> float:
    # 对每个 r 的轨道分别检测周期，比较是否一致；以一致为“正样本”计算F1
    if true_trajs.size == 0 or pred_trajs.size == 0:
        return float("nan")
    n = min(true_trajs.shape[0], pred_trajs.shape[0])
    y_true = []
    y_pred = []
    for i in range(n):
        pt = detect_period_from_traj(true_trajs[i], max_p=max_p, tol=tol)
        pp = detect_period_from_traj(pred_trajs[i], max_p=max_p, tol=tol)
        # 定义：预测周期等于真实周期（含共同为0）为1，否则为0
        y_true.append(1)
        y_pred.append(1 if pt == pp else 0)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_pred = normalize_rows_nonnegative(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    js_list = [js_divergence(y_true[i], y_pred[i]) for i in range(y_true.shape[0])]
    mean_js = float(np.mean(js_list)) if js_list else float("nan")
    return float(mse), mean_js

def evaluate_additional_metrics(
    densities_true: np.ndarray,
    densities_pred: np.ndarray,
    bin_edges: np.ndarray,
    traj_true: np.ndarray,
    traj_pred: np.ndarray,
    period_max: int = 8,
    period_tol: float = 0.02,
) -> dict:
    ssim_val = ssim_global(densities_true, densities_pred)
    emd_val = emd_1d_rows(densities_true, densities_pred, bin_edges=bin_edges)
    f1 = period_consistency_f1(traj_true, traj_pred, max_p=period_max, tol=period_tol)
    return {"ssim": ssim_val, "emd": emd_val, "period_f1": f1}

def export_per_r_bucket_csv(
    r_vals: np.ndarray,
    dens_true: np.ndarray,
    dens_pred: np.ndarray,
    bin_edges: np.ndarray,
    bucket_bins: int,
    out_csv: str = "per_r_metrics.csv",
) -> None:
    # 每个 r 行计算 MSE/JS/EMD，写CSV，并按 r 分桶汇总追加到文件末尾
    rows = []
    for i in range(dens_true.shape[0]):
        y_t = dens_true[i:i+1]
        y_p = dens_pred[i:i+1]
        mse_i, js_i = evaluate_predictions(y_t, y_p)
        emd_i = emd_1d_rows(y_t, y_p, bin_edges)
        rows.append((float(r_vals[i]), mse_i, js_i, emd_i))
    try:
        with open(out_csv, "w") as f:
            f.write("r,mse,js,emd\n")
            for r, mse, js, emd in rows:
                f.write(f"{r:.8f},{mse:.8e},{js:.8e},{emd:.8e}\n")
            # 分桶
            edges = np.linspace(min(r_vals), max(r_vals), num=max(bucket_bins, 1) + 1)
            f.write("\n[buckets]\n")
            f.write("bucket_left,bucket_right,count,mean_mse,mean_js,mean_emd\n")
            for b in range(len(edges) - 1):
                l, r = edges[b], edges[b+1]
                idxs = [k for k,(rv,_,_,_) in enumerate(rows) if rv >= l and (rv < r or (b == len(edges)-2 and rv <= r))]
                if not idxs:
                    f.write(f"{l:.6f},{r:.6f},0,NA,NA,NA\n")
                    continue
                mse_mean = float(np.mean([rows[k][1] for k in idxs]))
                js_mean = float(np.mean([rows[k][2] for k in idxs]))
                emd_mean = float(np.mean([rows[k][3] for k in idxs]))
                f.write(f"{l:.6f},{r:.6f},{len(idxs)},{mse_mean:.6e},{js_mean:.6e},{emd_mean:.6e}\n")
    except Exception:
        pass


def compute_lyapunov_for_r_values(
    r_values: np.ndarray,
    num_iterations: int,
    num_discard: int,
    seed: int,
    progress: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lyap = np.zeros_like(r_values, dtype=np.float64)
    iterator = enumerate(tqdm(r_values, desc="lyapunov (r)", disable=not progress))
    for idx, r in iterator:
        x0 = rng.uniform(0.1, 0.9)
        xs = simulate_logistic_series(r, num_iterations=num_iterations, num_discard=num_discard, x0=x0)
        if xs.size == 0:
            lyap[idx] = np.nan
            continue
        deriv = np.abs(r * (1.0 - 2.0 * xs))
        deriv = np.clip(deriv, 1e-12, None)
        lyap[idx] = float(np.mean(np.log(deriv)))
    return lyap


def moving_average_ignore_nan(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr.copy()
    w = np.ones(window, dtype=np.float64)
    valid = np.convolve((~np.isnan(arr)).astype(np.float64), w, mode="same")
    summed = np.convolve(np.nan_to_num(arr, nan=0.0), w, mode="same")
    valid[valid == 0] = 1.0
    return summed / valid


def auto_detect_split_r(
    r_values: np.ndarray,
    lyap: np.ndarray,
    eps: float = 1e-4,
    run: int = 5,
    smooth: int = 5,
    default_value: float = 3.56994,
) -> float:
    """根据 Lyapunov 指数自动检测混沌起点。
    策略：对 lyap 做平滑后，寻找首个长度为 run 的连续正段（>eps）。
    若未找到，返回 default_value。
    """
    if r_values.size == 0 or lyap.size == 0 or r_values.size != lyap.size:
        return float(default_value)
    y = moving_average_ignore_nan(lyap.astype(np.float64), int(max(1, smooth)))
    positive = y > float(eps)
    n = positive.size
    if n < run:
        return float(default_value)
    # 窗口和法快速检测连续 run 个 True
    kernel = np.ones(run, dtype=np.int32)
    pos_int = positive.astype(np.int32)
    conv = np.convolve(pos_int, kernel, mode="valid")
    idxs = np.where(conv == run)[0]
    if idxs[0:1].size > 0:
        return float(r_values[idxs[0]])
    return float(default_value)


def plot_lyapunov_curve(r_values: np.ndarray, lyap: np.ndarray, split_r: float, out_path: str) -> None:
    plt.figure(figsize=(9, 4), dpi=120)
    plt.plot(r_values, lyap, lw=1.2, label="Lyapunov")
    plt.axhline(0.0, color="k", lw=1.0, ls=":")
    plt.axvline(split_r, color="crimson", ls="--", lw=1.0)
    plt.xlabel("r")
    plt.ylabel("lambda")
    plt.title("Lyapunov exponent vs r")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)


def build_ts_dataset(
    r_values: np.ndarray,
    num_iterations: int,
    num_discard: int,
    seed: int,
    window: int,
    horizon: int,
    max_windows_per_r: int,
    progress: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    iterator = tqdm(r_values, desc="build ts dataset (r)", disable=not progress)
    for r in iterator:
        x0 = rng.uniform(0.1, 0.9)
        xs = simulate_logistic_series(r, num_iterations=num_iterations, num_discard=num_discard, x0=x0)
        T = xs.shape[0]
        if T <= window or T - window < horizon:
            continue
        start_max = T - window - horizon
        if start_max <= 0:
            continue
        count = min(max_windows_per_r, start_max)
        if count < start_max:
            starts = np.linspace(0, start_max - 1, num=count, dtype=int)
        else:
            starts = np.arange(0, start_max, dtype=int)
        for s in starts:
            win = xs[s:s + window]
            fut = xs[s + window:s + window + horizon]
            X_list.append(np.concatenate([[r], win]))
            y_list.append(fut)
    if not X_list:
        return np.empty((0, window + 1)), np.empty((0, horizon))
    X = np.vstack(X_list)
    Y = np.vstack(y_list)
    return X, Y


def simulate_with_ts_surrogate_iterative(
    model,
    r_values: np.ndarray,
    num_iterations: int,
    num_discard: int,
    window: int,
    seed: int,
    progress: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    num_r = r_values.shape[0]
    T_keep = max(0, num_iterations - num_discard)
    if T_keep == 0 or num_r == 0:
        return np.zeros((num_r, 0), dtype=np.float64)
    x_init = rng.uniform(0.1, 0.9, size=num_r)
    win = np.zeros((num_r, window), dtype=np.float64)
    for k in range(window):
        x_init = np.clip(r_values * x_init * (1.0 - x_init), 0.0, 1.0)
        win[:, k] = x_init
    kept = np.zeros((num_r, T_keep), dtype=np.float64)
    keep_idx = 0
    iterator = tqdm(range(num_iterations), desc="simulate via ts", disable=not progress)
    for t in iterator:
        X = np.column_stack([r_values, win])
        x_next = model.predict(X)
        if isinstance(x_next, np.ndarray) and x_next.ndim > 1:
            x_next = x_next[:, 0]
        x_next = np.clip(x_next, 0.0, 1.0)
        win = np.roll(win, shift=-1, axis=1)
        win[:, -1] = x_next
        if t >= num_discard:
            kept[:, keep_idx] = x_next
            keep_idx += 1
    return kept


def plot_ts_selected_compare(
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_examples: int,
    out_path: str,
) -> None:
    if X_test.shape[0] == 0:
        return
    idxs = np.linspace(0, X_test.shape[0] - 1, num=min(num_examples, X_test.shape[0]), dtype=int)
    plt.figure(figsize=(4 * min(3, len(idxs)), 3 * int(math.ceil(len(idxs) / 3))), dpi=120)
    for i, idx in enumerate(idxs, start=1):
        plt.subplot(int(math.ceil(len(idxs) / 3)), min(3, len(idxs)), i)
        plt.plot(y_true[idx], label="true", lw=1.2)
        plt.plot(y_pred[idx], label="pred", lw=1.2)
        plt.title(f"r = {X_test[idx, 0]:.4f}")
        if i == 1:
            plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


def plot_bifurcation_scatter(r_points: np.ndarray, x_points: np.ndarray, split_r: float, out_path: str) -> None:
    plt.figure(figsize=(9, 6), dpi=120)
    plt.scatter(r_points, x_points, s=0.1, c="#1f77b4", alpha=0.6)
    plt.axvline(split_r, color="crimson", ls="--", lw=1.2, label=f"split r = {split_r:.3f}")
    plt.xlabel("r")
    plt.ylabel("x")
    plt.title("Logistic Map Bifurcation (scatter)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)


def plot_bifurcation_scatter_pred(
    r_train: np.ndarray,
    x_train: np.ndarray,
    r_test: np.ndarray,
    x_test: np.ndarray,
    split_r: float,
    out_path: str,
) -> None:
    plt.figure(figsize=(9, 6), dpi=120)
    # 不做区域过滤，直接绘制传入的训练与预测散点，避免在不同方向（有序→混沌 / 混沌→有序）下出现空图
    if r_train.size:
        plt.scatter(r_train, x_train, s=0.1, c="#1f77b4", alpha=0.6, label="train")
    if r_test.size:
        plt.scatter(r_test, x_test, s=0.1, c="crimson", alpha=0.5, label="pred")
    plt.axvline(split_r, color="crimson", ls="--", lw=1.2)
    plt.xlabel("r")
    plt.ylabel("x")
    plt.title("Predicted Bifurcation (scatter)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)


def plot_density_heatmap(
    r_values: np.ndarray,
    bin_edges: np.ndarray,
    densities: np.ndarray,
    title: str,
    out_path: str,
) -> None:
    extent = [float(r_values.min()), float(r_values.max()), 0.0, 1.0]
    plt.figure(figsize=(9, 6), dpi=120)
    plt.imshow(
        densities.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="plasma",
        interpolation="nearest",
    )
    plt.colorbar(label="density")
    plt.xlabel("r")
    plt.ylabel("x")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)


def plot_compare_selected_r(
    r_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bin_edges: np.ndarray,
    num_examples: int,
    out_path: str,
) -> None:
    if r_test.shape[0] == 0:
        return
    idxs = np.linspace(0, r_test.shape[0] - 1, num=min(num_examples, r_test.shape[0]), dtype=int)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    cols = min(3, len(idxs))
    rows = int(math.ceil(len(idxs) / cols))
    plt.figure(figsize=(4 * cols, 3 * rows), dpi=120)
    for i, idx in enumerate(idxs, start=1):
        plt.subplot(rows, cols, i)
        plt.plot(centers, y_true[idx], label="true", lw=1.2)
        plt.plot(centers, y_pred[idx], label="pred", lw=1.2)
        plt.title(f"r = {r_test[idx, 0]:.4f}")
        plt.xlabel("x")
        plt.ylabel("density")
        if i == 1:
            plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)


# =====================
# GRU + MLP 组合模型
# =====================

class GRUMLPModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, mlp_hidden: Tuple[int, ...], output_size: int, residual: bool = True, r_feat_dim: int = 1, mlp_dropout: float = 0.0, spectral_norm: bool = False):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        layers: List[nn.Module] = []
        in_dim = hidden_size + r_feat_dim  # 拼接 r 特征
        for h in mlp_hidden:
            lin = nn.Linear(in_dim, h)
            if spectral_norm and (torch is not None):
                try:
                    from torch.nn.utils.parametrizations import spectral_norm as _sn
                    lin = _sn(lin)
                except Exception:
                    pass
            layers.append(lin)
            layers.append(nn.ReLU())
            if mlp_dropout and mlp_dropout > 0.0:
                layers.append(nn.Dropout(p=mlp_dropout))
            in_dim = h
        out_lin = nn.Linear(in_dim, output_size)
        if spectral_norm and (torch is not None):
            try:
                from torch.nn.utils.parametrizations import spectral_norm as _sn
                out_lin = _sn(out_lin)
            except Exception:
                pass
        layers.append(out_lin)
        self.mlp = nn.Sequential(*layers)
        self.horizon = output_size
        self.residual = residual

    def forward(self, seq_x: torch.Tensor, r_feat: torch.Tensor, r_scalar: torch.Tensor) -> torch.Tensor:
        # seq_x: [B, T, D], r_feat: [B, R], r_scalar: [B, 1]
        _, h_T = self.gru(seq_x)  # [num_layers, B, H]
        h_last = h_T[-1]  # [B, H]
        x = torch.cat([h_last, r_feat], dim=1)
        mlp_out = self.mlp(x)
        if not self.residual:
            return torch.clip(mlp_out, 0.0, 1.0)
        # 物理先验：基线为逻辑映射从窗口末值开始的 H 步外推
        last_x = seq_x[:, -1:, 0]  # [B, 1] 使用第一通道视为原始 x
        base_steps = []
        cur = last_x
        for _ in range(self.horizon):
            cur = r_scalar * cur * (1.0 - cur)
            base_steps.append(cur)
        base = torch.cat(base_steps, dim=1)  # [B, H]
        out = base + mlp_out
        return torch.clip(out, 0.0, 1.0)


class GRUMLPRegressor:
    def __init__(
        self,
        window: int,
        horizon: int,
        hidden_size: int,
        mlp_hidden: Tuple[int, ...],
        num_layers: int = 1,
        dropout: float = 0.0,
        lr: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 300,
        seed: int = 42,
        device: Optional[str] = None,
        residual: bool = True,
        residual_lambda: float = 1e-3,
        poly_degree: int = 1,
        input_jitter: float = 0.0,
        grad_clip: float = 1.0,
        r_min: float = 2.5,
        r_max: float = 4.0,
        r_poly_degree: int = 1,
        r_sin_cos: bool = False,
        r_norm_to_unit: bool = False,
        weight_decay: float = 0.0,
        rollout_k: int = 0,
        rollout_lambda: float = 0.0,
        mlp_dropout: float = 0.0,
        spectral_norm: bool = False,
    ) -> None:
        if torch is None:
            raise RuntimeError("需要安装 PyTorch 才能使用 GRU 模型：pip install torch")
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
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.residual = residual
        self.residual_lambda = residual_lambda
        self.poly_degree = max(1, int(poly_degree))
        self.input_jitter = float(input_jitter)
        self.grad_clip = float(grad_clip)
        self.r_min = r_min
        self.r_max = r_max
        self.r_poly_degree = max(1, int(r_poly_degree))
        self.r_sin_cos = bool(r_sin_cos)
        self.r_norm_to_unit = bool(r_norm_to_unit)
        self.weight_decay = float(weight_decay)
        self.rollout_k = int(rollout_k)
        self.rollout_lambda = float(rollout_lambda)
        r_feat_dim = self._r_feat_dim()
        self.model = GRUMLPModel(input_size=self.poly_degree, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, mlp_hidden=mlp_hidden, output_size=horizon, residual=residual, r_feat_dim=r_feat_dim, mlp_dropout=mlp_dropout, spectral_norm=spectral_norm).to(self.device)
        self.loss_fn = nn.MSELoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _to_tensors(self, X: np.ndarray, Y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # X: [N, 1 + window], 前 1 列为 r
        r = X[:, 0:1].astype(np.float32)
        seq = X[:, 1:].astype(np.float32)
        y = Y.astype(np.float32)
        # 归一保证在 [0,1] 已由上游生成
        # 多项式序列特征
        if self.poly_degree > 1:
            feats = [seq]
            for p in range(2, self.poly_degree + 1):
                feats.append(np.clip(np.power(seq, p), 0.0, 1.0))
            seq_multi = np.stack(feats, axis=-1)  # [N, T, D]
            seq_t = torch.from_numpy(seq_multi)
        else:
            seq_t = torch.from_numpy(seq).unsqueeze(-1)  # [N, T, 1]
        r_scalar_t = torch.from_numpy(r)  # [N, 1]
        # r 特征
        r_feat = self._build_r_features(r)
        r_feat_t = torch.from_numpy(r_feat.astype(np.float32))
        y_t = torch.from_numpy(y)  # [N, H]
        return seq_t, r_feat_t, r_scalar_t, y_t

    def _r_feat_dim(self) -> int:
        dim = self.r_poly_degree
        if self.r_sin_cos:
            dim += 2
        return dim

    def _build_r_features(self, r_np: np.ndarray) -> np.ndarray:
        # r_np: [N,1]
        if self.r_norm_to_unit:
            # 线性映射到 [-1,1]
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

    @staticmethod
    def _compute_base_from_window(seq_b: torch.Tensor, r_b: torch.Tensor, horizon: int) -> torch.Tensor:
        # seq_b: [B, T, D] 或 [B, T, 1]；基线基于原始 x，取第一特征
        if seq_b.dim() == 3 and seq_b.size(-1) > 1:
            last_x = seq_b[:, -1:, 0]
        else:
            last_x = seq_b[:, -1:, 0]
        steps = []
        cur = last_x
        for _ in range(horizon):
            cur = r_b * cur * (1.0 - cur)
            steps.append(cur)
        return torch.cat(steps, dim=1)

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False) -> None:
        if X.shape[0] == 0:
            return
        seq_t, r_feat_t, r_scalar_t, y_t = self._to_tensors(X, Y)
        # 输入扰动（训练增强）
        if self.input_jitter > 0:
            noise = torch.randn_like(seq_t) * self.input_jitter
            seq_t = torch.clamp(seq_t + noise, 0.0, 1.0)
        ds = TensorDataset(seq_t, r_feat_t, r_scalar_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        best_loss = float("inf")
        patience = 20
        since_best = 0
        for epoch in range(self.max_epochs):
            loss_sum = 0.0
            count = 0
            for seq_b, r_feat_b, r_scalar_b, y_b in dl:
                seq_b = seq_b.to(self.device)
                r_feat_b = r_feat_b.to(self.device)
                r_scalar_b = r_scalar_b.to(self.device)
                y_b = y_b.to(self.device)
                self.opt.zero_grad()
                pred = self.model(seq_b, r_feat_b, r_scalar_b)
                loss = self.loss_fn(pred, y_b)
                if self.residual and self.residual_lambda > 0.0:
                    base = self._compute_base_from_window(seq_b, r_scalar_b, self.model.horizon)
                    res = pred - base
                    loss = loss + self.residual_lambda * torch.mean(res * res)
                # 物理滚动一致性（无监督）：K 步自由滚动与逻辑基线对齐
                if self.rollout_k and self.rollout_lambda > 0.0:
                    # 准备窗口副本
                    win = seq_b.clone()
                    roll_loss = 0.0
                    for _ in range(self.rollout_k):
                        step_pred = self.model(win, r_feat_b, r_scalar_b)
                        # 只取第一步（iter 训练）
                        if step_pred.dim() == 2:
                            step_next = step_pred[:, 0:1]
                        else:
                            step_next = step_pred[:, 0, 0:1]
                        base_next = r_scalar_b * win[:, -1:, 0] * (1.0 - win[:, -1:, 0])
                        roll_loss = roll_loss + torch.mean((step_next - base_next) ** 2)
                        # 更新窗口
                        if win.dim() == 3 and win.size(-1) > 1:
                            # 仅第一通道替换为预测，其他多项式通道按幂次重建
                            x_new = step_next  # [B,1]
                            x_new3 = x_new.unsqueeze(-1)  # [B,1,1]
                            win = torch.roll(win, shifts=-1, dims=1)
                            win[:, -1:, 0:1] = x_new3
                            for p in range(2, self.poly_degree + 1):
                                win[:, -1:, p-1:p] = torch.clamp((x_new ** p).unsqueeze(-1), 0.0, 1.0)
                        else:
                            win = torch.roll(win, shifts=-1, dims=1)
                            win[:, -1:, 0:1] = step_next.unsqueeze(-1)
                    loss = loss + (self.rollout_lambda * roll_loss / float(self.rollout_k))
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
                loss_sum += float(loss.detach().cpu()) * seq_b.size(0)
                count += seq_b.size(0)
            epoch_loss = loss_sum / max(count, 1)
            # 先更新早停状态
            improved = epoch_loss + 1e-8 < best_loss
            if improved:
                best_loss = epoch_loss
                since_best = 0
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
            else:
                since_best += 1
                if since_best >= patience:
                    if verbose:
                        logging.getLogger("bifurcation").info("[GRU] early stopping")
                    break
            # 再打印更丰富的日志（每 10 个 epoch 或最后一个 epoch）
            if verbose and ((epoch % 10 == 0) or (epoch == self.max_epochs - 1)):
                lr_cur = float(self.opt.param_groups[0].get('lr', 0.0))
                logging.getLogger("bifurcation").info(
                    f"[GRU] epoch {epoch+1}/{self.max_epochs} | "
                    f"loss={epoch_loss:.6e} | lr={lr_cur:.2e} | patience={since_best}/{patience}"
                )
        # 恢复最佳
        if 'best_state' in locals():
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

    # -------- 垃圾神经元分析与剪枝（针对首层 MLP 输入的 GRU 隐状态列） --------
    def analyze_gru_hidden_neuron_importance(self) -> np.ndarray:
        layer0 = None
        for m in self.model.mlp:
            if isinstance(m, nn.Linear) or hasattr(m, 'weight'):
                layer0 = m
                break
        if layer0 is None:
            raise RuntimeError("未找到 MLP 首层 Linear")
        W = layer0.weight  # [out, in]
        in_dim = W.shape[1]
        hidden_dim = self.hidden_size
        if in_dim < hidden_dim:
            hidden_dim = in_dim
        W_gru = W[:, :hidden_dim]
        imp = torch.mean(torch.abs(W_gru), dim=0).detach().cpu().numpy()  # [hidden_dim]
        return imp

    def prune_garbage_neurons(self, threshold: float, max_neurons: int) -> List[int]:
        imp = self.analyze_gru_hidden_neuron_importance()
        # 选择重要性低于阈值的索引；最多删除 max_neurons 个
        cand = np.where(imp < float(threshold))[0]
        if cand.size == 0:
            return []
        order = np.argsort(imp[cand])  # 从小到大
        to_prune = cand[order][:max(0, int(max_neurons))]
        if to_prune.size == 0:
            return []
        # 对应首层 Linear 的前 hidden_size 列置零（等价删除）
        layer0 = None
        for m in self.model.mlp:
            if isinstance(m, nn.Linear) or hasattr(m, 'weight'):
                layer0 = m
                break
        W = layer0.weight
        with torch.no_grad():
            for j in to_prune:
                if j < W.shape[1]:
                    W[:, j].zero_()
        logging.getLogger("bifurcation").info(f"[prune] pruned {len(to_prune)} neurons below threshold={threshold}")
        return to_prune.tolist()

    def finetune_after_prune(self, X: np.ndarray, Y: np.ndarray, epochs: int, verbose: bool = False) -> None:
        if epochs <= 0:
            return
        # 轻微微调
        seq_t, r_feat_t, r_scalar_t, y_t = self._to_tensors(X, Y)
        ds = TensorDataset(seq_t, r_feat_t, r_scalar_t, y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for epoch in range(epochs):
            loss_sum = 0.0
            count = 0
            for seq_b, r_feat_b, r_scalar_b, y_b in dl:
                seq_b = seq_b.to(self.device)
                r_feat_b = r_feat_b.to(self.device)
                r_scalar_b = r_scalar_b.to(self.device)
                y_b = y_b.to(self.device)
                self.opt.zero_grad()
                pred = self.model(seq_b, r_feat_b, r_scalar_b)
                loss = self.loss_fn(pred, y_b)
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
                loss_sum += float(loss.detach().cpu()) * seq_b.size(0)
                count += seq_b.size(0)
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                logging.getLogger("bifurcation").info(f"[prune-ft] epoch {epoch+1}/{epochs}, loss={loss_sum/max(count,1):.6e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] == 0:
            return np.empty((0, self.horizon), dtype=np.float32)
        self.model.eval()
        seq_t, r_feat_t, r_scalar_t, _ = self._to_tensors(X, np.zeros((X.shape[0], self.horizon), dtype=np.float32))
        preds: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, X.shape[0], 4096):
                seq_b = seq_t[i:i+4096].to(self.device)
                r_feat_b = r_feat_t[i:i+4096].to(self.device)
                r_scalar_b = r_scalar_t[i:i+4096].to(self.device)
                out = self.model(seq_b, r_feat_b, r_scalar_b)
                preds.append(out.detach().cpu().numpy())
        pred = np.concatenate(preds, axis=0)
        return pred


class CurriculumGRUMLPRegressor(GRUMLPRegressor):
    """带课程学习的 GRU-MLP 回归器"""

    def __init__(self, *args, use_curriculum: bool = True, curriculum_stages: int = 3, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_curriculum = use_curriculum
        self.curriculum_stages = max(1, int(curriculum_stages))

    def _compute_rollout_loss(self, seq_b: torch.Tensor, r_feat_b: torch.Tensor, r_scalar_b: torch.Tensor) -> torch.Tensor:
        win = seq_b.clone()
        roll_loss = 0.0
        for _ in range(self.rollout_k):
            step_pred = self.model(win, r_feat_b, r_scalar_b)
            if step_pred.dim() == 2:
                step_next = step_pred[:, 0:1]
            else:
                step_next = step_pred[:, 0, 0:1]
            base_next = r_scalar_b * win[:, -1:, 0] * (1.0 - win[:, -1:, 0])
            roll_loss = roll_loss + torch.mean((step_next - base_next) ** 2)
            # 更新窗口
            if win.dim() == 3 and win.size(-1) > 1:
                x_new3 = step_next.unsqueeze(-1)
                win = torch.roll(win, shifts=-1, dims=1)
                win[:, -1:, 0:1] = x_new3
                for p in range(2, self.poly_degree + 1):
                    win[:, -1:, p-1:p] = torch.clamp((step_next ** p).unsqueeze(-1), 0.0, 1.0)
            else:
                win = torch.roll(win, shifts=-1, dims=1)
                win[:, -1:, 0:1] = step_next.unsqueeze(-1)
        return roll_loss / float(max(self.rollout_k, 1))

    def fit_with_curriculum(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False) -> None:
        if X.shape[0] == 0:
            return
        if not self.use_curriculum:
            return self.fit(X, Y, verbose=verbose)
        logger = logging.getLogger("bifurcation")
        seq_t, r_feat_t, r_scalar_t, y_t = self._to_tensors(X, Y)
        if self.input_jitter > 0:
            noise = torch.randn_like(seq_t) * self.input_jitter
            seq_t = torch.clamp(seq_t + noise, 0.0, 1.0)
        r_values = X[:, 0]
        sorted_indices = np.argsort(r_values)
        stage_fracs = np.linspace(0.3, 1.0, self.curriculum_stages)
        epochs_per_stage = max(1, self.max_epochs // self.curriculum_stages)
        best_loss = float('inf')
        patience_counter = 0
        patience = 25
        for stage_idx, frac in enumerate(stage_fracs):
            n_samples = int(len(sorted_indices) * frac)
            stage_indices = sorted_indices[:max(1, n_samples)]
            r_min_stage = float(r_values[stage_indices].min())
            r_max_stage = float(r_values[stage_indices].max())
            logger.info("=" * 50)
            logger.info(f"Curriculum Stage {stage_idx+1}/{self.curriculum_stages}: {n_samples}/{len(X)} ({frac*100:.1f}%), r in [{r_min_stage:.5f}, {r_max_stage:.5f}]")
            logger.info("=" * 50)
            stage_ds = TensorDataset(seq_t[stage_indices], r_feat_t[stage_indices], r_scalar_t[stage_indices], y_t[stage_indices])
            stage_dl = DataLoader(stage_ds, batch_size=self.batch_size, shuffle=True)
            self.model.train()
            for epoch in range(epochs_per_stage):
                loss_sum = 0.0
                count = 0
                for seq_b, r_feat_b, r_scalar_b, y_b in stage_dl:
                    seq_b = seq_b.to(self.device)
                    r_feat_b = r_feat_b.to(self.device)
                    r_scalar_b = r_scalar_b.to(self.device)
                    y_b = y_b.to(self.device)
                    self.opt.zero_grad()
                    pred = self.model(seq_b, r_feat_b, r_scalar_b)
                    loss = self.loss_fn(pred, y_b)
                    if self.residual and self.residual_lambda > 0.0:
                        base = self._compute_base_from_window(seq_b, r_scalar_b, self.model.horizon)
                        res = pred - base
                        loss = loss + self.residual_lambda * torch.mean(res * res)
                    if self.rollout_k and self.rollout_lambda > 0.0:
                        roll_loss = self._compute_rollout_loss(seq_b, r_feat_b, r_scalar_b)
                        loss = loss + self.rollout_lambda * roll_loss
                    loss.backward()
                    if self.grad_clip and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.opt.step()
                    loss_sum += float(loss.detach().cpu()) * seq_b.size(0)
                    count += seq_b.size(0)
                epoch_loss = loss_sum / max(count, 1)
                if verbose and (epoch % 20 == 0):
                    logger.info(f"  Epoch {epoch+1}/{epochs_per_stage}: loss={epoch_loss:.6e}")
                if stage_idx == len(stage_fracs) - 1:
                    if epoch_loss < best_loss - 1e-7:
                        best_loss = epoch_loss
                        patience_counter = 0
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info("  Early stopping")
                            break
            logger.info(f"Stage {stage_idx+1} done, loss={epoch_loss:.6e}")
        if 'best_state' in locals():
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
            logger.info(f"Restored best model, loss={best_loss:.6e}")


def parse_hidden_layers(hidden_str: str) -> Tuple[int, ...]:
    if not hidden_str:
        return ExperimentConfig.hidden_layers
    layers = tuple(int(part.strip()) for part in hidden_str.split(",") if part.strip())
    return layers or ExperimentConfig.hidden_layers


def build_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    hidden_layers = parse_hidden_layers(args.hidden)
    return ExperimentConfig(
        r_min=args.r_min,
        r_max=args.r_max,
        num_r=args.num_r,
        split_r=args.split_r,
        num_iterations=args.num_iterations,
        num_discard=args.num_discard,
        num_bins=args.num_bins,
        seed=args.seed,
        hidden_layers=hidden_layers,
        max_train_iter=args.max_train_iter,
        learning_rate_init=args.lr,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        pca_fit_on=args.pca_fit_on,
        model_type=args.model,
        symmetrize=args.symmetrize,
        smooth_sigma=args.smooth_sigma,
        scatter_per_r=args.scatter_per_r,
        mode=args.mode,
        map_pairs_per_r=args.map_pairs_per_r,
        progress=args.progress,
        mlp_verbose=args.verbose,
        ts_window=args.ts_window,
        ts_horizon=args.ts_horizon,
        ts_strategy=args.ts_strategy,
        ts_max_windows_per_r=args.ts_max_windows_per_r,
        lyapunov=args.lyapunov,
        gru_hidden_size=args.gru_hidden_size,
        gru_num_layers=args.gru_num_layers,
        gru_dropout=args.gru_dropout,
        gru_batch_size=args.gru_batch_size,
        train_side=args.train_side,
        gru_residual=args.gru_residual,
        gru_residual_lambda=args.gru_residual_lambda,
        gru_poly_degree=args.gru_poly_degree,
        ts_input_jitter=args.ts_input_jitter,
        grad_clip=args.grad_clip,
        gru_r_poly_degree=args.gru_r_poly_degree,
        gru_r_sin_cos=args.gru_r_sin_cos,
        r_norm_to_unit=args.r_norm_to_unit,
        gru_weight_decay=args.gru_weight_decay,
        rollout_consistency_k=args.rollout_consistency_k,
        rollout_consistency_lambda=args.rollout_consistency_lambda,
        mlp_dropout=args.mlp_dropout,
        mlp_spectral_norm=args.mlp_spectral_norm,
        ts_phys_aug_sparse=args.ts_phys_aug_sparse,
        ts_phys_aug_rmin=args.ts_phys_aug_rmin,
        ts_phys_aug_rmax=args.ts_phys_aug_rmax,
        ts_phys_aug_per_r=args.ts_phys_aug_per_r,
        ts_phys_aug_mode=args.ts_phys_aug_mode,
        ts_phys_band=args.ts_phys_band,
        use_curriculum=args.use_curriculum,
        curriculum_stages=args.curriculum_stages,
        auto_split=args.auto_split,
        auto_split_run=args.auto_split_run,
        auto_split_eps=args.auto_split_eps,
        auto_split_smooth=args.auto_split_smooth,
        per_r_csv=args.per_r_csv,
        per_r_bucket_bins=args.per_r_bucket_bins,
        esn_reservoir=args.esn_reservoir,
        esn_spectral_radius=args.esn_spectral_radius,
        esn_leak=args.esn_leak,
        esn_alpha=args.esn_alpha,
        conv_channels=args.conv_channels,
        conv_kernel=args.conv_kernel,
        conv_dropout=args.conv_dropout,
        run_both=args.run_both,
    )


def build_train_test_masks(cfg: ExperimentConfig, r_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if cfg.train_side == "sparse":
        train_mask = r_values <= cfg.split_r
    else:
        train_mask = r_values > cfg.split_r
    return train_mask, ~train_mask


def prepare_base_artifacts(cfg: ExperimentConfig) -> BaseArtifacts:
    logger = logging.getLogger("bifurcation")
    r_values = np.linspace(cfg.r_min, cfg.r_max, cfg.num_r)

    logger.info("[stage] generate true bifurcation scatter ...")
    r_points, x_points = generate_bifurcation_data(
        r_values=r_values,
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        seed=cfg.seed,
        progress=cfg.progress,
    )

    logger.info("[stage] build true density matrix ...")
    densities_true, bin_edges = build_density_matrix(
        r_values=r_values,
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        num_bins=cfg.num_bins,
        seed=cfg.seed,
        progress=cfg.progress,
    )
    if cfg.symmetrize:
        densities_true = symmetrize_densities(densities_true)
    if cfg.smooth_sigma > 0.0:
        densities_true = smooth_along_r(densities_true, cfg.smooth_sigma)

    if cfg.lyapunov:
        logger.info("[stage] compute Lyapunov exponent ...")
        lyap = compute_lyapunov_for_r_values(
            r_values=r_values,
            num_iterations=cfg.num_iterations,
            num_discard=cfg.num_discard,
            seed=cfg.seed,
            progress=cfg.progress,
        )
        if cfg.auto_split:
            detected = auto_detect_split_r(
                r_values=r_values,
                lyap=lyap,
                eps=cfg.auto_split_eps,
                run=cfg.auto_split_run,
                smooth=cfg.auto_split_smooth,
                default_value=cfg.split_r,
            )
            if abs(float(detected) - float(cfg.split_r)) > 1e-6:
                logger.info(f"[auto_split] split_r {cfg.split_r:.5f} -> {detected:.5f}")
                cfg.split_r = float(detected)
        plot_lyapunov_curve(r_values, lyap, split_r=cfg.split_r, out_path="lyapunov_true.png")

    plot_bifurcation_scatter(r_points, x_points, split_r=cfg.split_r, out_path="bifurcation_scatter.png")
    plot_density_heatmap(
        r_values=r_values,
        bin_edges=bin_edges,
        densities=densities_true,
        title="True density heatmap",
        out_path="density_true.png",
    )
    return BaseArtifacts(r_values=r_values, densities_true=densities_true, bin_edges=bin_edges)


def convert_trajs_to_density(trajs: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    if trajs.size == 0:
        return np.zeros((trajs.shape[0], bin_edges.size - 1), dtype=np.float64)
    densities = np.zeros((trajs.shape[0], bin_edges.size - 1), dtype=np.float64)
    for i in range(trajs.shape[0]):
        hist, _ = np.histogram(trajs[i], bins=bin_edges, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float64)
        hist /= max(hist.sum(), 1.0)
        densities[i] = hist
    return densities


def build_true_trajs_for_r_values(r_values: np.ndarray, cfg: ExperimentConfig, target_len: int) -> np.ndarray:
    if target_len <= 0 or r_values.size == 0:
        return np.zeros((r_values.size, 0), dtype=np.float64)
    trajs = []
    for r in r_values:
        xs = simulate_logistic_series(r, num_iterations=cfg.num_iterations, num_discard=cfg.num_discard, x0=0.314)
        if xs.shape[0] >= target_len:
            trajs.append(xs[-target_len:])
        else:
            pad = target_len - xs.shape[0]
            trajs.append(np.pad(xs, (pad, 0), mode="edge"))
    return np.stack(trajs, axis=0)


def maybe_write_metrics_csv(metrics: dict, filename: str = "metrics_extra.csv") -> None:
    try:
        with open(filename, "w") as f:
            f.write("metric,value\n")
            for key, value in metrics.items():
                f.write(f"{key},{value}\n")
    except Exception:
        pass


def maybe_export_per_r_metrics(
    cfg: ExperimentConfig,
    r_values_test: np.ndarray,
    y_true_eval: np.ndarray,
    y_pred_eval: np.ndarray,
    bin_edges: np.ndarray,
) -> None:
    if not cfg.per_r_csv:
        return
    try:
        export_per_r_bucket_csv(
            r_vals=r_values_test,
            dens_true=y_true_eval,
            dens_pred=y_pred_eval,
            bin_edges=bin_edges,
            bucket_bins=cfg.per_r_bucket_bins,
            out_csv="per_r_metrics.csv",
        )
    except Exception:
        pass


def log_test_metrics(tag: str, mse: float, js: float) -> None:
    logger = logging.getLogger("bifurcation")
    logger.info(f"{tag} Test MSE: {mse:.6e}\n{tag} Test mean-JS: {js:.6e}")


def run_map_mode(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    artifacts: BaseArtifacts,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    logger = logging.getLogger("bifurcation")
    if not np.any(train_mask):
        logger.warning("Map 模式缺少训练区域，已跳过。")
        return
    if not np.any(test_mask):
        logger.warning("Map 模式缺少测试区域，已跳过。")
        return
    logger.info("[stage] build (r,x)->x_next training pairs ...")
    X_pairs, y_pairs = build_map_training_data(
        r_values=artifacts.r_values[train_mask],
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        seed=cfg.seed,
        max_pairs_per_r=cfg.map_pairs_per_r,
        progress=cfg.progress,
    )
    if X_pairs.shape[0] == 0:
        logger.warning("Map 模式没有可用的训练样本，已跳过。")
        return
    print(f"[stage] train map MLP, pairs={X_pairs.shape[0]:,} ...", flush=True)
    map_model = train_map_model(
        X_train=X_pairs,
        y_train=y_pairs,
        hidden_layers=cfg.hidden_layers,
        max_iter=cfg.max_train_iter,
        learning_rate_init=cfg.learning_rate_init,
        seed=cfg.seed,
        verbose=cfg.mlp_verbose,
    )
    logger.info("[stage] simulate trajectories via learned map ...")
    kept = simulate_with_surrogate_vectorized(
        model=map_model,
        r_values=artifacts.r_values[test_mask],
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        seed=cfg.seed + 7,
        progress=cfg.progress,
    )
    take = min(cfg.scatter_per_r, kept.shape[1]) if kept.size else 0
    if take > 0:
        sel = np.arange(kept.shape[1] - take, kept.shape[1])
        x_pred = kept[:, sel].reshape(-1)
        r_rep = np.repeat(artifacts.r_values[test_mask], take)
    else:
        x_pred = np.array([])
        r_rep = np.array([])
    r_train_scatter, x_train_scatter = sample_scatter_from_densities(
        r_values=artifacts.r_values[train_mask],
        bin_edges=artifacts.bin_edges,
        densities=artifacts.densities_true[train_mask],
        samples_per_r=cfg.scatter_per_r,
        seed=cfg.seed,
    )
    plot_bifurcation_scatter_pred(
        r_train=r_train_scatter,
        x_train=x_train_scatter,
        r_test=r_rep,
        x_test=x_pred,
        split_r=cfg.split_r,
        out_path="bifurcation_scatter_pred.png",
    )
    pred_full = artifacts.densities_true.copy()
    if kept.size > 0:
        pred_full[test_mask] = convert_trajs_to_density(kept, artifacts.bin_edges)
    pred_full = normalize_rows_nonnegative(pred_full)
    plot_density_heatmap(
        r_values=artifacts.r_values,
        bin_edges=artifacts.bin_edges,
        densities=pred_full,
        title="Predicted density heatmap via learned map",
        out_path="density_pred.png",
    )
    y_true_eval = artifacts.densities_true[test_mask]
    y_pred_eval = pred_full[test_mask]
    mse, mean_js = evaluate_predictions(y_true=y_true_eval, y_pred=y_pred_eval)
    log_test_metrics("[map]", mse, mean_js)
    if args.eval_extra:
        true_trajs = build_true_trajs_for_r_values(
            artifacts.r_values[test_mask],
            cfg,
            kept.shape[1] if kept.size else 0,
        )
        extra = evaluate_additional_metrics(
            densities_true=y_true_eval,
            densities_pred=y_pred_eval,
            bin_edges=artifacts.bin_edges,
            traj_true=true_trajs,
            traj_pred=kept,
            period_max=args.period_max,
            period_tol=args.period_tol,
        )
        maybe_write_metrics_csv(extra)
    maybe_export_per_r_metrics(cfg, artifacts.r_values[test_mask], y_true_eval, y_pred_eval, artifacts.bin_edges)


def run_density_mode(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    artifacts: BaseArtifacts,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> None:
    logger = logging.getLogger("bifurcation")
    if not np.any(train_mask):
        logger.warning("Density 模式缺少训练区域，已跳过。")
        return
    if not np.any(test_mask):
        logger.warning("Density 模式缺少测试区域，已跳过。")
        return
    X_all = artifacts.r_values.reshape(-1, 1)
    X_train, X_test = X_all[train_mask], X_all[test_mask]
    if cfg.use_pca:
        pca = PCA(n_components=cfg.pca_components, svd_solver="auto")
        if cfg.pca_fit_on == "all":
            pca.fit(artifacts.densities_true)
        else:
            pca.fit(artifacts.densities_true[train_mask])
        y_train = pca.transform(artifacts.densities_true[train_mask])
        model = train_regressor(
            X_train=X_train,
            y_train=y_train,
            model_type=cfg.model_type if cfg.model_type in ("mlp", "ridge") else "mlp",
            hidden_layers=cfg.hidden_layers,
            max_iter=cfg.max_train_iter,
            learning_rate_init=cfg.learning_rate_init,
            seed=cfg.seed,
            verbose=cfg.mlp_verbose,
        )
        y_pred = pca.inverse_transform(model.predict(X_test))
    else:
        y_train = artifacts.densities_true[train_mask]
        model = train_regressor(
            X_train=X_train,
            y_train=y_train,
            model_type=cfg.model_type if cfg.model_type in ("mlp", "ridge") else "mlp",
            hidden_layers=cfg.hidden_layers,
            max_iter=cfg.max_train_iter,
            learning_rate_init=cfg.learning_rate_init,
            seed=cfg.seed,
            verbose=cfg.mlp_verbose,
        )
        y_pred = model.predict(X_test)
    y_true_eval = artifacts.densities_true[test_mask]
    mse, mean_js = evaluate_predictions(y_true=y_true_eval, y_pred=y_pred)
    log_test_metrics("[density]", mse, mean_js)
    pred_full = artifacts.densities_true.copy()
    pred_full[test_mask] = y_pred
    pred_full = normalize_rows_nonnegative(pred_full)
    plot_density_heatmap(
        r_values=artifacts.r_values,
        bin_edges=artifacts.bin_edges,
        densities=pred_full,
        title="Predicted density heatmap (sparse->dense)",
        out_path="density_pred.png",
    )
    plot_compare_selected_r(
        r_test=X_test,
        y_true=y_true_eval,
        y_pred=y_pred,
        bin_edges=artifacts.bin_edges,
        num_examples=args.examples,
        out_path="density_selected_compare.png",
    )
    if args.eval_extra:
        extra = {
            "ssim": ssim_global(y_true_eval, y_pred),
            "emd": emd_1d_rows(y_true_eval, y_pred, artifacts.bin_edges),
            "period_f1": float("nan"),
        }
        maybe_write_metrics_csv(extra)
    maybe_export_per_r_metrics(cfg, artifacts.r_values[test_mask], y_true_eval, y_pred, artifacts.bin_edges)


def apply_ts_phys_aug(
    cfg: ExperimentConfig,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not cfg.ts_phys_aug_sparse or cfg.ts_strategy != "iter":
        return X_tr, Y_tr
    logger = logging.getLogger("bifurcation")
    if cfg.ts_phys_aug_mode == "auto":
        if cfg.train_side == "sparse":
            r_aug_min = max(cfg.r_min, cfg.split_r - cfg.ts_phys_band)
            r_aug_max = cfg.split_r
        else:
            r_aug_min = cfg.split_r
            r_aug_max = min(cfg.r_max, cfg.split_r + cfg.ts_phys_band)
        logger.info(f"[phys-aug] auto band [{r_aug_min:.5f}, {r_aug_max:.5f}] around split={cfg.split_r:.5f}")
    else:
        if cfg.train_side == "sparse":
            r_aug_min = float(np.clip(cfg.ts_phys_aug_rmin, cfg.r_min, cfg.split_r))
            r_aug_max = float(np.clip(cfg.ts_phys_aug_rmax, cfg.r_min, cfg.split_r))
        else:
            r_aug_min = float(np.clip(cfg.ts_phys_aug_rmin, cfg.split_r, cfg.r_max))
            r_aug_max = float(np.clip(cfg.ts_phys_aug_rmax, cfg.split_r, cfg.r_max))
        if r_aug_max < r_aug_min:
            r_aug_min, r_aug_max = r_aug_max, r_aug_min
        logger.info(f"[phys-aug] manual band [{r_aug_min:.5f}, {r_aug_max:.5f}]")
    if r_aug_max <= r_aug_min + 1e-12:
        return X_tr, Y_tr
    r_aug_values = np.linspace(r_aug_min, r_aug_max, num=40)
    X_aug, Y_aug = build_ts_dataset(
        r_values=r_aug_values,
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        seed=seed,
        window=cfg.ts_window,
        horizon=1,
        max_windows_per_r=cfg.ts_phys_aug_per_r,
        progress=False,
    )
    if X_aug.shape[0] > 0:
        X_tr = np.vstack([X_tr, X_aug])
        Y_tr = np.vstack([Y_tr, Y_aug])
    return X_tr, Y_tr


def fit_ts_model(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    horizon: int,
    model_seed: int,
):
    if cfg.model_type == "gru":
        if torch is None:
            raise RuntimeError("当前环境未安装 torch，无法使用 --model gru。请先安装：pip install torch")
        print(f"[stage] train GRU-MLP, train={X_tr.shape[0]:,} windows ...", flush=True)
        gru_epochs = max(1, min(cfg.max_train_iter, 400))
        ModelCls = CurriculumGRUMLPRegressor if cfg.use_curriculum else GRUMLPRegressor
        ts_model = ModelCls(
            window=cfg.ts_window,
            horizon=horizon,
            hidden_size=cfg.gru_hidden_size,
            mlp_hidden=cfg.hidden_layers,
            num_layers=cfg.gru_num_layers,
            dropout=cfg.gru_dropout,
            lr=cfg.learning_rate_init,
            batch_size=cfg.gru_batch_size,
            max_epochs=gru_epochs,
            seed=model_seed,
            residual=cfg.gru_residual,
            residual_lambda=cfg.gru_residual_lambda,
            poly_degree=cfg.gru_poly_degree,
            input_jitter=cfg.ts_input_jitter,
            grad_clip=cfg.grad_clip,
            r_min=cfg.r_min,
            r_max=cfg.r_max,
            r_poly_degree=cfg.gru_r_poly_degree,
            r_sin_cos=cfg.gru_r_sin_cos,
            r_norm_to_unit=cfg.r_norm_to_unit,
            weight_decay=cfg.gru_weight_decay,
            rollout_k=cfg.rollout_consistency_k,
            rollout_lambda=cfg.rollout_consistency_lambda,
            mlp_dropout=cfg.mlp_dropout,
            spectral_norm=cfg.mlp_spectral_norm,
        )
        if cfg.use_curriculum:
            ts_model.fit_with_curriculum(X_tr, Y_tr, verbose=cfg.mlp_verbose)
        else:
            ts_model.fit(X_tr, Y_tr, verbose=cfg.mlp_verbose)
        if cfg.prune_after_train:
            pruned = ts_model.prune_garbage_neurons(threshold=cfg.prune_threshold, max_neurons=cfg.prune_max_neurons)
            if pruned:
                ts_model.finetune_after_prune(X_tr, Y_tr, epochs=cfg.prune_retrain_epochs, verbose=cfg.mlp_verbose)
        return ts_model
    print(f"[stage] train TS regressor ({cfg.model_type}), train={X_tr.shape[0]:,} windows ...", flush=True)
    return train_regressor(
        X_train=X_tr,
        y_train=Y_tr,
        model_type=cfg.model_type,
        hidden_layers=cfg.hidden_layers,
        max_iter=cfg.max_train_iter,
        learning_rate_init=cfg.learning_rate_init,
        seed=model_seed,
        verbose=cfg.mlp_verbose,
    )


def execute_ts_direction(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    artifacts: BaseArtifacts,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    seeds: dict,
    make_plots: bool,
    log_prefix: str,
) -> dict:
    logger = logging.getLogger("bifurcation")
    if not np.any(train_mask) or not np.any(test_mask):
        logger.warning(f"{log_prefix} 缺少训练或测试区域，已跳过。")
        return {"mse": float("nan"), "js": float("nan"), "extra": {"ssim": float("nan"), "emd": float("nan"), "period_f1": float("nan")}}
    tr_h = cfg.ts_horizon if cfg.ts_strategy == "direct" else 1
    te_h = cfg.ts_horizon if cfg.ts_strategy == "direct" else 1
    seed_base_train = seeds.get("train_ds", cfg.seed)
    seed_base_test = seeds.get("test_ds", cfg.seed + 1)
    seed_phys = seeds.get("phys_aug", cfg.seed + 11)
    seed_sim = seeds.get("simulate", cfg.seed + 7)
    seed_model = seeds.get("model", cfg.seed)
    X_tr, Y_tr = build_ts_dataset(
        r_values=artifacts.r_values[train_mask],
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        seed=seed_base_train,
        window=cfg.ts_window,
        horizon=tr_h,
        max_windows_per_r=cfg.ts_max_windows_per_r,
        progress=cfg.progress,
    )
    X_tr, Y_tr = apply_ts_phys_aug(cfg, X_tr, Y_tr, seed_phys)
    X_te, Y_te = build_ts_dataset(
        r_values=artifacts.r_values[test_mask],
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        seed=seed_base_test,
        window=cfg.ts_window,
        horizon=te_h,
        max_windows_per_r=min(cfg.ts_max_windows_per_r, 400),
        progress=cfg.progress,
    )
    ts_model = fit_ts_model(cfg, args, X_tr, Y_tr, tr_h, seed_model)
    if X_te.shape[0] > 0:
        Y_pred_te = ts_model.predict(X_te)
        mse_ts = mean_squared_error(Y_te, Y_pred_te)
        logger.info(f"{log_prefix} Window-forecast MSE: {mse_ts:.6e}")
        if make_plots:
            plot_ts_selected_compare(
                X_test=X_te,
                y_true=Y_te,
                y_pred=Y_pred_te,
                num_examples=args.examples,
                out_path="ts_selected_compare.png",
            )
    if cfg.ts_strategy != "iter":
        return {"mse": float("nan"), "js": float("nan"), "extra": {"ssim": float("nan"), "emd": float("nan"), "period_f1": float("nan")}}
    logger.info("[stage] simulate trajectories via TS surrogate ...")
    kept_ts = simulate_with_ts_surrogate_iterative(
        model=ts_model,
        r_values=artifacts.r_values[test_mask],
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        window=cfg.ts_window,
        seed=seed_sim,
        progress=cfg.progress,
    )
    take = min(cfg.scatter_per_r, kept_ts.shape[1]) if kept_ts.size else 0
    if take > 0:
        sel = np.arange(kept_ts.shape[1] - take, kept_ts.shape[1])
        x_pred = kept_ts[:, sel].reshape(-1)
        r_rep = np.repeat(artifacts.r_values[test_mask], take)
    else:
        x_pred = np.array([])
        r_rep = np.array([])
    if make_plots:
        r_train_scatter, x_train_scatter = sample_scatter_from_densities(
            r_values=artifacts.r_values[train_mask],
            bin_edges=artifacts.bin_edges,
            densities=artifacts.densities_true[train_mask],
            samples_per_r=cfg.scatter_per_r,
            seed=cfg.seed,
        )
        plot_bifurcation_scatter_pred(
            r_train=r_train_scatter,
            x_train=x_train_scatter,
            r_test=r_rep,
            x_test=x_pred,
            split_r=cfg.split_r,
            out_path="bifurcation_scatter_pred.png",
        )
    pred_full = artifacts.densities_true.copy()
    if kept_ts.size > 0:
        pred_full[test_mask] = convert_trajs_to_density(kept_ts, artifacts.bin_edges)
    pred_full = normalize_rows_nonnegative(pred_full)
    if make_plots:
        plot_density_heatmap(
            r_values=artifacts.r_values,
            bin_edges=artifacts.bin_edges,
            densities=pred_full,
            title="Predicted density heatmap via TS surrogate",
            out_path="density_pred.png",
        )
    y_true_eval = artifacts.densities_true[test_mask]
    y_pred_eval = pred_full[test_mask]
    mse, mean_js = evaluate_predictions(y_true=y_true_eval, y_pred=y_pred_eval)
    log_test_metrics(log_prefix, mse, mean_js)
    metrics_extra = {"ssim": float("nan"), "emd": float("nan"), "period_f1": float("nan")}
    if args.eval_extra:
        true_trajs = build_true_trajs_for_r_values(
            artifacts.r_values[test_mask],
            cfg,
            kept_ts.shape[1] if kept_ts.ndim > 1 else 0,
        )
        metrics_extra = evaluate_additional_metrics(
            densities_true=y_true_eval,
            densities_pred=y_pred_eval,
            bin_edges=artifacts.bin_edges,
            traj_true=true_trajs,
            traj_pred=kept_ts,
            period_max=args.period_max,
            period_tol=args.period_tol,
        )
        if make_plots:
            maybe_write_metrics_csv(metrics_extra)
    if make_plots:
        maybe_export_per_r_metrics(cfg, artifacts.r_values[test_mask], y_true_eval, y_pred_eval, artifacts.bin_edges)
    return {"mse": mse, "js": mean_js, "extra": metrics_extra}


def main():
def main():
    parser = argparse.ArgumentParser(description="使用密集区训练模型预测对数映射密度/轨道（支持 GRU-MLP 组合）")
    parser.add_argument("--r_min", type=float, default=ExperimentConfig.r_min)
    parser.add_argument("--r_max", type=float, default=ExperimentConfig.r_max)
    parser.add_argument("--num_r", type=int, default=ExperimentConfig.num_r)
    parser.add_argument("--split_r", type=float, default=ExperimentConfig.split_r)
    parser.add_argument("--num_iterations", type=int, default=ExperimentConfig.num_iterations)
    parser.add_argument("--num_discard", type=int, default=ExperimentConfig.num_discard)
    parser.add_argument("--num_bins", type=int, default=ExperimentConfig.num_bins)
    parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
    parser.add_argument("--hidden", type=str, default="256,256", help="MLP隐藏层，如 128,128,64")
    parser.add_argument("--max_train_iter", type=int, default=ExperimentConfig.max_train_iter)
    parser.add_argument("--lr", type=float, default=ExperimentConfig.learning_rate_init)
    parser.add_argument("--examples", type=int, default=6, help="逐r对比曲线示例数")
    parser.add_argument("--no_show", action="store_true", help="仅保存图片不显示窗口")
    parser.add_argument("--model", type=str, choices=["mlp", "ridge", "svr", "gru"], default="mlp")
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--pca_components", type=int, default=16)
    parser.add_argument("--pca_fit_on", type=str, choices=["sparse", "all"], default="sparse")
    parser.add_argument("--symmetrize", action="store_true")
    parser.add_argument("--smooth_sigma", type=float, default=0.0, help="沿r方向高斯平滑sigma，0为不平滑")
    parser.add_argument("--scatter_per_r", type=int, default=200, help="预测散点图每个r采样数")
    parser.add_argument("--mode", type=str, choices=["map", "density", "ts"], default="map")
    parser.add_argument("--map_pairs_per_r", type=int, default=1200)
    # 时间序列相关
    parser.add_argument("--ts_window", type=int, default=16)
    parser.add_argument("--ts_horizon", type=int, default=16)
    parser.add_argument("--ts_strategy", type=str, choices=["direct", "iter"], default="iter")
    parser.add_argument("--ts_max_windows_per_r", type=int, default=800)
    # Lyapunov
    parser.add_argument("--lyapunov", dest="lyapunov", action="store_true", default=True)
    parser.add_argument("--no_lyapunov", dest="lyapunov", action="store_false")
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    # 自动 split 检测
    parser.add_argument("--auto_split", action="store_true", default=False, help="基于 Lyapunov 自动检测 split_r")
    parser.add_argument("--auto_split_run", type=int, default=5, help="判定连续为正的最小长度")
    parser.add_argument("--auto_split_eps", type=float, default=1e-4, help="正阈值 eps")
    parser.add_argument("--auto_split_smooth", type=int, default=5, help="Lyapunov 平滑窗口")
    # GRU 相关
    parser.add_argument("--gru_hidden_size", type=int, default=ExperimentConfig.gru_hidden_size)
    parser.add_argument("--gru_num_layers", type=int, default=ExperimentConfig.gru_num_layers)
    parser.add_argument("--gru_dropout", type=float, default=ExperimentConfig.gru_dropout)
    parser.add_argument("--gru_batch_size", type=int, default=ExperimentConfig.gru_batch_size)
    # 训练区选择
    parser.add_argument("--train_side", type=str, choices=["sparse", "dense"], default="sparse", help="训练区：sparse 使用 r<=split_r；dense 使用 r>split_r")
    # 物理先验与增强
    parser.add_argument("--gru_residual", dest="gru_residual", action="store_true", default=True, help="GRU 输出为对逻辑映射的残差")
    parser.add_argument("--no_gru_residual", dest="gru_residual", action="store_false", help="关闭残差头，直接回归值")
    parser.add_argument("--gru_residual_lambda", type=float, default=ExperimentConfig.gru_residual_lambda, help="残差 L2 正则系数")
    parser.add_argument("--gru_poly_degree", type=int, default=ExperimentConfig.gru_poly_degree, help="序列多项式特征阶数，1 表示仅 x")
    parser.add_argument("--ts_input_jitter", type=float, default=ExperimentConfig.ts_input_jitter, help="训练时窗口输入高斯噪声标准差")
    parser.add_argument("--grad_clip", type=float, default=ExperimentConfig.grad_clip, help="梯度裁剪阈值，0 关闭")
    # r 特征与优化
    parser.add_argument("--gru_r_poly_degree", type=int, default=ExperimentConfig.gru_r_poly_degree)
    parser.add_argument("--gru_r_sin_cos", action="store_true", default=ExperimentConfig.gru_r_sin_cos)
    parser.add_argument("--r_norm_to_unit", action="store_true", default=ExperimentConfig.r_norm_to_unit)
    parser.add_argument("--gru_weight_decay", type=float, default=ExperimentConfig.gru_weight_decay)
    # 物理滚动一致性
    parser.add_argument("--rollout_consistency_k", type=int, default=ExperimentConfig.rollout_consistency_k)
    parser.add_argument("--rollout_consistency_lambda", type=float, default=ExperimentConfig.rollout_consistency_lambda)
    # 稀疏区物理合成数据增强
    parser.add_argument("--ts_phys_aug_sparse", action="store_true", default=ExperimentConfig.ts_phys_aug_sparse)
    parser.add_argument("--ts_phys_aug_rmin", type=float, default=ExperimentConfig.ts_phys_aug_rmin)
    parser.add_argument("--ts_phys_aug_rmax", type=float, default=ExperimentConfig.ts_phys_aug_rmax)
    parser.add_argument("--ts_phys_aug_per_r", type=int, default=ExperimentConfig.ts_phys_aug_per_r)
    parser.add_argument("--ts_phys_aug_mode", type=str, choices=["auto","manual"], default=ExperimentConfig.ts_phys_aug_mode)
    parser.add_argument("--ts_phys_band", type=float, default=ExperimentConfig.ts_phys_band, help="auto 模式下围绕 split_r 的增强带宽")
    # MLP 正则化
    parser.add_argument("--mlp_spectral_norm", action="store_true", default=ExperimentConfig.mlp_spectral_norm)
    parser.add_argument("--mlp_dropout", type=float, default=ExperimentConfig.mlp_dropout)
    # 日志
    parser.add_argument("--log_file", type=str, default=None, help="将日志写入文件（可选）")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"]) 
    # 剪枝
    parser.add_argument("--prune_after_train", action="store_true", default=False, help="训练后执行垃圾神经元剪枝并微调")
    parser.add_argument("--prune_threshold", type=float, default=0.09, help="剪枝阈值（均值绝对权重）")
    parser.add_argument("--prune_max_neurons", type=int, default=4, help="每次最多删除的神经元数")
    parser.add_argument("--prune_retrain_epochs", type=int, default=60, help="剪枝后微调 epoch")
    # 课程学习
    parser.add_argument("--use_curriculum", dest="use_curriculum", action="store_true", default=True, help="启用课程学习")
    parser.add_argument("--no_curriculum", dest="use_curriculum", action="store_false", help="关闭课程学习")
    parser.add_argument("--curriculum_stages", type=int, default=3, help="课程学习阶段数")
    # 扩展评估
    parser.add_argument("--eval_extra", action="store_true", default=False, help="计算SSIM/EMD/周期一致F1并保存CSV")
    parser.add_argument("--period_max", type=int, default=8)
    parser.add_argument("--period_tol", type=float, default=0.02)
    # 逐r与分桶导出
    parser.add_argument("--per_r_csv", action="store_true", default=False)
    parser.add_argument("--per_r_bucket_bins", type=int, default=4)
    # ESN baseline
    parser.add_argument("--esn_reservoir", type=int, default=500)
    parser.add_argument("--esn_spectral_radius", type=float, default=0.9)
    parser.add_argument("--esn_leak", type=float, default=1.0)
    parser.add_argument("--esn_alpha", type=float, default=1e-4)
    # Conv1D baseline
    parser.add_argument("--conv_channels", type=int, default=64)
    parser.add_argument("--conv_kernel", type=int, default=3)
    parser.add_argument("--conv_dropout", type=float, default=0.1)
    # 自动双向
    parser.add_argument("--run_both", action="store_true", default=False, help="同参双向运行并导出LaTeX表")

    args = parser.parse_args()
    logger = setup_logging(args.log_file, args.log_level)

    hidden_layers = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    cfg = ExperimentConfig(
        r_min=args.r_min,
        r_max=args.r_max,
        num_r=args.num_r,
        split_r=args.split_r,
        num_iterations=args.num_iterations,
        num_discard=args.num_discard,
        num_bins=args.num_bins,
        seed=args.seed,
        hidden_layers=hidden_layers,
        max_train_iter=args.max_train_iter,
        learning_rate_init=args.lr,
        use_pca=args.use_pca,
        pca_components=args.pca_components,
        pca_fit_on=args.pca_fit_on,
        model_type=args.model,
        symmetrize=args.symmetrize,
        smooth_sigma=args.smooth_sigma,
        scatter_per_r=args.scatter_per_r,
        mode=args.mode,
        map_pairs_per_r=args.map_pairs_per_r,
        progress=args.progress,
        mlp_verbose=args.verbose,
        ts_window=args.ts_window,
        ts_horizon=args.ts_horizon,
        ts_strategy=args.ts_strategy,
        ts_max_windows_per_r=args.ts_max_windows_per_r,
        lyapunov=args.lyapunov,
        gru_hidden_size=args.gru_hidden_size,
        gru_num_layers=args.gru_num_layers,
        gru_dropout=args.gru_dropout,
        gru_batch_size=args.gru_batch_size,
        train_side=args.train_side,
        gru_residual=args.gru_residual,
        gru_residual_lambda=args.gru_residual_lambda,
        gru_poly_degree=args.gru_poly_degree,
        ts_input_jitter=args.ts_input_jitter,
        grad_clip=args.grad_clip,
        gru_r_poly_degree=args.gru_r_poly_degree,
        gru_r_sin_cos=args.gru_r_sin_cos,
        r_norm_to_unit=args.r_norm_to_unit,
        gru_weight_decay=args.gru_weight_decay,
        rollout_consistency_k=args.rollout_consistency_k,
        rollout_consistency_lambda=args.rollout_consistency_lambda,
        mlp_dropout=args.mlp_dropout,
        mlp_spectral_norm=args.mlp_spectral_norm,
        ts_phys_aug_sparse=args.ts_phys_aug_sparse,
        ts_phys_aug_rmin=args.ts_phys_aug_rmin,
        ts_phys_aug_rmax=args.ts_phys_aug_rmax,
        ts_phys_aug_per_r=args.ts_phys_aug_per_r,
        ts_phys_aug_mode=args.ts_phys_aug_mode,
        ts_phys_band=args.ts_phys_band,
        use_curriculum=args.use_curriculum,
        curriculum_stages=args.curriculum_stages,
        auto_split=args.auto_split,
        auto_split_run=args.auto_split_run,
        auto_split_eps=args.auto_split_eps,
        auto_split_smooth=args.auto_split_smooth,
        per_r_csv=args.per_r_csv,
        per_r_bucket_bins=args.per_r_bucket_bins,
        esn_reservoir=args.esn_reservoir,
        esn_spectral_radius=args.esn_spectral_radius,
        esn_leak=args.esn_leak,
        esn_alpha=args.esn_alpha,
        conv_channels=args.conv_channels,
        conv_kernel=args.conv_kernel,
        conv_dropout=args.conv_dropout,
        run_both=args.run_both,
    )

    rng = np.random.default_rng(cfg.seed)
    r_values = np.linspace(cfg.r_min, cfg.r_max, cfg.num_r)

    # 1) 真实分岔散点
    logging.getLogger("bifurcation").info("[stage] generate true bifurcation scatter ...")
    r_points, x_points = generate_bifurcation_data(
        r_values=r_values,
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        seed=cfg.seed,
        progress=cfg.progress,
    )
    plot_bifurcation_scatter(r_points, x_points, split_r=cfg.split_r, out_path="bifurcation_scatter.png")

    # 2) 真实密度
    logging.getLogger("bifurcation").info("[stage] build true density matrix ...")
    densities_true, bin_edges = build_density_matrix(
        r_values=r_values,
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        num_bins=cfg.num_bins,
        seed=cfg.seed,
        progress=cfg.progress,
    )
    if cfg.symmetrize:
        densities_true = symmetrize_densities(densities_true)
    if cfg.smooth_sigma > 0.0:
        densities_true = smooth_along_r(densities_true, cfg.smooth_sigma)
    plot_density_heatmap(
        r_values=r_values,
        bin_edges=bin_edges,
        densities=densities_true,
        title="True density heatmap",
        out_path="density_true.png",
    )

    if cfg.lyapunov:
        logging.getLogger("bifurcation").info("[stage] compute Lyapunov exponent ...")
        lyap = compute_lyapunov_for_r_values(
            r_values=r_values,
            num_iterations=cfg.num_iterations,
            num_discard=cfg.num_discard,
            seed=cfg.seed,
            progress=cfg.progress,
        )
        # 可选：根据 Lyapunov 自动检测混沌起点
        if cfg.auto_split:
            detected = auto_detect_split_r(
                r_values=r_values,
                lyap=lyap,
                eps=cfg.auto_split_eps,
                run=cfg.auto_split_run,
                smooth=cfg.auto_split_smooth,
                default_value=cfg.split_r,
            )
            if abs(float(detected) - float(cfg.split_r)) > 1e-6:
                logging.getLogger("bifurcation").info(f"[auto_split] split_r {cfg.split_r:.5f} -> {detected:.5f}")
                cfg.split_r = float(detected)
        plot_lyapunov_curve(r_values, lyap, split_r=cfg.split_r, out_path="lyapunov_true.png")

    # 3) 模式分支
    if cfg.train_side == "sparse":
        train_mask = r_values <= cfg.split_r
        test_mask = ~train_mask
    else:
        train_mask = r_values > cfg.split_r
        test_mask = ~train_mask

    if cfg.mode == "map":
        logging.getLogger("bifurcation").info("[stage] build (r,x)->x_next training pairs ...")
        X_pairs, y_pairs = build_map_training_data(
            r_values=r_values[train_mask],
            num_iterations=cfg.num_iterations,
            num_discard=cfg.num_discard,
            seed=cfg.seed,
            max_pairs_per_r=cfg.map_pairs_per_r,
            progress=cfg.progress,
        )
        print(f"[stage] train map MLP, pairs={X_pairs.shape[0]:,} ...", flush=True)
        map_model = train_map_model(
            X_train=X_pairs,
            y_train=y_pairs,
            hidden_layers=cfg.hidden_layers,
            max_iter=cfg.max_train_iter,
            learning_rate_init=cfg.learning_rate_init,
            seed=cfg.seed,
            verbose=cfg.mlp_verbose,
        )
        logging.getLogger("bifurcation").info("[stage] simulate trajectories via learned map ...")
        kept = simulate_with_surrogate_vectorized(
            model=map_model,
            r_values=r_values[test_mask],
            num_iterations=cfg.num_iterations,
            num_discard=cfg.num_discard,
            seed=cfg.seed + 7,
            progress=cfg.progress,
        )
        T_keep = kept.shape[1]
        take = min(cfg.scatter_per_r, T_keep)
        if take > 0:
            sel = np.arange(T_keep - take, T_keep)
            x_pred = kept[:, sel].reshape(-1)
            r_rep = np.repeat(r_values[test_mask], take)
        else:
            x_pred = np.array([])
            r_rep = np.array([])
        r_train_scatter, x_train_scatter = sample_scatter_from_densities(
            r_values=r_values[train_mask],
            bin_edges=bin_edges,
            densities=densities_true[train_mask],
            samples_per_r=cfg.scatter_per_r,
            seed=cfg.seed,
        )
        plot_bifurcation_scatter_pred(
            r_train=r_train_scatter,
            x_train=x_train_scatter,
            r_test=r_rep,
            x_test=x_pred,
            split_r=cfg.split_r,
            out_path="bifurcation_scatter_pred.png",
        )
        pred_full = densities_true.copy()
        if kept.size > 0:
            pred_dense = np.zeros_like(densities_true[test_mask])
            for i in range(kept.shape[0]):
                hist, _ = np.histogram(kept[i], bins=bin_edges, range=(0.0, 1.0), density=False)
                hist = hist.astype(np.float64)
                hist /= max(hist.sum(), 1.0)
                pred_dense[i] = hist
            pred_full[test_mask] = pred_dense
        pred_full = normalize_rows_nonnegative(pred_full)
        logging.getLogger("bifurcation").info("[stage] render predicted density heatmap (map mode) ...")
        plot_density_heatmap(
            r_values=r_values,
            bin_edges=bin_edges,
            densities=pred_full,
            title="Predicted density heatmap via learned map",
            out_path="density_pred.png",
        )
        y_true_eval = densities_true[test_mask]
        y_pred_eval = pred_full[test_mask]
        mse, mean_js = evaluate_predictions(y_true=y_true_eval, y_pred=y_pred_eval)
        logging.getLogger("bifurcation").info(f"[map] Test MSE: {mse:.6e}\n[map] Test mean-JS: {mean_js:.6e}")
        if args.eval_extra:
            true_trajs = []
            if kept.size > 0:
                for r in r_values[test_mask]:
                    xs = simulate_logistic_series(r, num_iterations=cfg.num_iterations, num_discard=cfg.num_discard, x0=0.314)
                    true_trajs.append(xs[-kept.shape[1]:] if xs.shape[0] >= kept.shape[1] else np.pad(xs, (kept.shape[1]-xs.shape[0],0), mode="edge"))
                true_trajs = np.stack(true_trajs, axis=0) if true_trajs else np.zeros_like(kept)
            else:
                true_trajs = np.zeros((y_true_eval.shape[0], 0), dtype=np.float64)
            extra = evaluate_additional_metrics(
                densities_true=y_true_eval,
                densities_pred=y_pred_eval,
                bin_edges=bin_edges,
                traj_true=true_trajs,
                traj_pred=kept,
                period_max=args.period_max,
                period_tol=args.period_tol,
            )
            try:
                with open("metrics_extra.csv", "w") as f:
                    f.write("metric,value\n")
                    for k,v in extra.items():
                        f.write(f"{k},{v}\n")
            except Exception:
                pass
        if cfg.per_r_csv:
            try:
                export_per_r_bucket_csv(
                    r_vals=r_values[test_mask],
                    dens_true=y_true_eval,
                    dens_pred=y_pred_eval,
                    bin_edges=bin_edges,
                    bucket_bins=cfg.per_r_bucket_bins,
                    out_csv="per_r_metrics.csv",
                )
            except Exception:
                pass

    elif cfg.mode == "density":
        X_all = r_values.reshape(-1, 1)
        X_train, X_test = X_all[train_mask], X_all[test_mask]
        if cfg.use_pca:
            pca = PCA(n_components=cfg.pca_components, svd_solver="auto")
            if cfg.pca_fit_on == "all":
                pca.fit(densities_true)
            else:
                pca.fit(densities_true[train_mask])
            y_train_coef = pca.transform(densities_true[train_mask])
            y_test_coef = pca.transform(densities_true[test_mask])
            logging.getLogger("bifurcation").info("[stage] train density regressor (PCA head) ...")
            model = train_regressor(
                X_train=X_train,
                y_train=y_train_coef,
                model_type=cfg.model_type if cfg.model_type in ("mlp", "ridge") else "mlp",
                hidden_layers=cfg.hidden_layers,
                max_iter=cfg.max_train_iter,
                learning_rate_init=cfg.learning_rate_init,
                seed=cfg.seed,
                verbose=cfg.mlp_verbose,
            )
            y_pred_test_coef = model.predict(X_test)
            y_pred_test = pca.inverse_transform(y_pred_test_coef)
        else:
            y_train = densities_true[train_mask]
            y_test = densities_true[test_mask]
            logging.getLogger("bifurcation").info("[stage] train density regressor ...")
            model = train_regressor(
                X_train=X_train,
                y_train=y_train,
                model_type=cfg.model_type if cfg.model_type in ("mlp", "ridge") else "mlp",
                hidden_layers=cfg.hidden_layers,
                max_iter=cfg.max_train_iter,
                learning_rate_init=cfg.learning_rate_init,
                seed=cfg.seed,
                verbose=cfg.mlp_verbose,
            )
            y_pred_test = model.predict(X_test)
        y_true_eval = densities_true[test_mask]
        mse, mean_js = evaluate_predictions(y_true=y_true_eval, y_pred=y_pred_test)
        logging.getLogger("bifurcation").info(f"[density] Test MSE: {mse:.6e}\n[density] Test mean-JS: {mean_js:.6e}")
        pred_full = densities_true.copy()
        pred_full[test_mask] = y_pred_test
        pred_full = normalize_rows_nonnegative(pred_full)
        logging.getLogger("bifurcation").info("[stage] render predicted density heatmap (density mode) ...")
        plot_density_heatmap(
            r_values=r_values,
            bin_edges=bin_edges,
            densities=pred_full,
            title="Predicted density heatmap (sparse->dense)",
            out_path="density_pred.png",
        )
        plot_compare_selected_r(
            r_test=X_test,
            y_true=y_true_eval,
            y_pred=y_pred_test,
            bin_edges=bin_edges,
            num_examples=args.examples,
            out_path="density_selected_compare.png",
        )
        if args.eval_extra:
            extra = {
                "ssim": ssim_global(y_true_eval, y_pred_test),
                "emd": emd_1d_rows(y_true_eval, y_pred_test, bin_edges),
                "period_f1": float("nan"),
            }
            try:
                with open("metrics_extra.csv", "w") as f:
                    f.write("metric,value\n")
                    for k,v in extra.items():
                        f.write(f"{k},{v}\n")
            except Exception:
                pass
        if cfg.per_r_csv:
            try:
                export_per_r_bucket_csv(
                    r_vals=X_test.reshape(-1),
                    dens_true=y_true_eval,
                    dens_pred=y_pred_test,
                    bin_edges=bin_edges,
                    bucket_bins=cfg.per_r_bucket_bins,
                    out_csv="per_r_metrics.csv",
                )
            except Exception:
                pass

    else:  # ts
        logging.getLogger("bifurcation").info("[stage] build time-series dataset ...")
        tr_h = cfg.ts_horizon if cfg.ts_strategy == "direct" else 1
        X_tr, Y_tr = build_ts_dataset(
            r_values=r_values[train_mask],
            num_iterations=cfg.num_iterations,
            num_discard=cfg.num_discard,
            seed=cfg.seed,
            window=cfg.ts_window,
            horizon=tr_h,
            max_windows_per_r=cfg.ts_max_windows_per_r,
            progress=cfg.progress,
        )
        # 稀疏/密集边界物理合成增强（auto: 围绕 split_r 的带宽；manual: 按用户范围）
        if cfg.ts_phys_aug_sparse and cfg.ts_strategy == "iter":
            if cfg.ts_phys_aug_mode == "auto":
                if cfg.train_side == "sparse":
                    r_aug_min = max(cfg.r_min, cfg.split_r - cfg.ts_phys_band)
                    r_aug_max = cfg.split_r
                else:
                    r_aug_min = cfg.split_r
                    r_aug_max = min(cfg.r_max, cfg.split_r + cfg.ts_phys_band)
                logging.getLogger("bifurcation").info(f"[phys-aug] auto band [{r_aug_min:.5f}, {r_aug_max:.5f}] around split={cfg.split_r:.5f}")
            else:
                if cfg.train_side == "sparse":
                    r_aug_min = float(np.clip(cfg.ts_phys_aug_rmin, cfg.r_min, cfg.split_r))
                    r_aug_max = float(np.clip(cfg.ts_phys_aug_rmax, cfg.r_min, cfg.split_r))
                else:
                    r_aug_min = float(np.clip(cfg.ts_phys_aug_rmin, cfg.split_r, cfg.r_max))
                    r_aug_max = float(np.clip(cfg.ts_phys_aug_rmax, cfg.split_r, cfg.r_max))
                if r_aug_max < r_aug_min:
                    r_aug_min, r_aug_max = r_aug_max, r_aug_min
                logging.getLogger("bifurcation").info(f"[phys-aug] manual band [{r_aug_min:.5f}, {r_aug_max:.5f}]")
            if r_aug_max > r_aug_min + 1e-12:
                r_aug_values = np.linspace(r_aug_min, r_aug_max, num=40)
                X_aug, Y_aug = build_ts_dataset(
                    r_values=r_aug_values,
                    num_iterations=cfg.num_iterations,
                    num_discard=cfg.num_discard,
                    seed=cfg.seed + 11,
                    window=cfg.ts_window,
                    horizon=tr_h,
                    max_windows_per_r=cfg.ts_phys_aug_per_r,
                    progress=False,
                )
                if X_aug.shape[0] > 0:
                    X_tr = np.vstack([X_tr, X_aug])
                    Y_tr = np.vstack([Y_tr, Y_aug])
        te_h = cfg.ts_horizon if cfg.ts_strategy == "direct" else 1
        X_te, Y_te = build_ts_dataset(
            r_values=r_values[test_mask],
            num_iterations=cfg.num_iterations,
            num_discard=cfg.num_discard,
            seed=cfg.seed + 1,
            window=cfg.ts_window,
            horizon=te_h,
            max_windows_per_r=min(cfg.ts_max_windows_per_r, 400),
            progress=cfg.progress,
        )

        if cfg.model_type == "gru":
            if torch is None:
                raise RuntimeError("当前环境未安装 torch，无法使用 --model gru。请先安装：pip install torch")
            print(f"[stage] train GRU-MLP, train={X_tr.shape[0]:,} windows ...", flush=True)
            gru_epochs = max(1, min(cfg.max_train_iter, 400))
            ModelCls = CurriculumGRUMLPRegressor if cfg.use_curriculum else GRUMLPRegressor
            ts_model = ModelCls(
                window=cfg.ts_window,
                horizon=tr_h,
                hidden_size=cfg.gru_hidden_size,
                mlp_hidden=cfg.hidden_layers,
                num_layers=cfg.gru_num_layers,
                dropout=cfg.gru_dropout,
                lr=cfg.learning_rate_init,
                batch_size=cfg.gru_batch_size,
                max_epochs=gru_epochs,
                seed=cfg.seed,
                residual=cfg.gru_residual,
                residual_lambda=cfg.gru_residual_lambda,
                poly_degree=cfg.gru_poly_degree,
                input_jitter=cfg.ts_input_jitter,
                grad_clip=cfg.grad_clip,
                r_min=cfg.r_min,
                r_max=cfg.r_max,
                r_poly_degree=cfg.gru_r_poly_degree,
                r_sin_cos=cfg.gru_r_sin_cos,
                r_norm_to_unit=cfg.r_norm_to_unit,
                weight_decay=cfg.gru_weight_decay,
                rollout_k=cfg.rollout_consistency_k,
                rollout_lambda=cfg.rollout_consistency_lambda,
                mlp_dropout=cfg.mlp_dropout,
                spectral_norm=cfg.mlp_spectral_norm,
            )
            if cfg.use_curriculum:
                ts_model.fit_with_curriculum(X_tr, Y_tr, verbose=cfg.mlp_verbose)
            else:
                ts_model.fit(X_tr, Y_tr, verbose=cfg.mlp_verbose)
            if cfg.prune_after_train:
                pruned = ts_model.prune_garbage_neurons(threshold=cfg.prune_threshold, max_neurons=cfg.prune_max_neurons)
                if pruned:
                    ts_model.finetune_after_prune(X_tr, Y_tr, epochs=cfg.prune_retrain_epochs, verbose=cfg.mlp_verbose)
        elif cfg.model_type in ("mlp","ridge","svr"):
            print(f"[stage] train TS regressor ({cfg.model_type}), train={X_tr.shape[0]:,} windows ...", flush=True)
            ts_model = train_regressor(
                X_train=X_tr,
                y_train=Y_tr,
                model_type=cfg.model_type,
                hidden_layers=cfg.hidden_layers,
                max_iter=cfg.max_train_iter,
                learning_rate_init=cfg.learning_rate_init,
                seed=cfg.seed,
                verbose=cfg.mlp_verbose,
            )
        else:
            raise RuntimeError(f"Unsupported ts model: {cfg.model_type}")

        if X_te.shape[0] > 0:
            Y_pred_te = ts_model.predict(X_te)
            mse_ts = mean_squared_error(Y_te, Y_pred_te)
            logging.getLogger("bifurcation").info(f"[ts] Window-forecast MSE: {mse_ts:.6e}")
            plot_ts_selected_compare(
                X_test=X_te,
                y_true=Y_te,
                y_pred=Y_pred_te,
                num_examples=args.examples,
                out_path="ts_selected_compare.png",
            )

        if cfg.ts_strategy == "iter":
            logging.getLogger("bifurcation").info("[stage] simulate trajectories via TS surrogate ...")
            kept_ts = simulate_with_ts_surrogate_iterative(
                model=ts_model,
                r_values=r_values[test_mask],
                num_iterations=cfg.num_iterations,
                num_discard=cfg.num_discard,
                window=cfg.ts_window,
                seed=cfg.seed + 7,
                progress=cfg.progress,
            )
            T_keep = kept_ts.shape[1]
            take = min(cfg.scatter_per_r, T_keep)
            if take > 0:
                sel = np.arange(T_keep - take, T_keep)
                x_pred = kept_ts[:, sel].reshape(-1)
                r_rep = np.repeat(r_values[test_mask], take)
            else:
                x_pred = np.array([])
                r_rep = np.array([])
            r_train_scatter, x_train_scatter = sample_scatter_from_densities(
                r_values=r_values[train_mask],
                bin_edges=bin_edges,
                densities=densities_true[train_mask],
                samples_per_r=cfg.scatter_per_r,
                seed=cfg.seed,
            )
            plot_bifurcation_scatter_pred(
                r_train=r_train_scatter,
                x_train=x_train_scatter,
                r_test=r_rep,
                x_test=x_pred,
                split_r=cfg.split_r,
                out_path="bifurcation_scatter_pred.png",
            )
            pred_full = densities_true.copy()
            if kept_ts.size > 0:
                pred_dense = np.zeros_like(densities_true[test_mask])
                for i in range(kept_ts.shape[0]):
                    hist, _ = np.histogram(kept_ts[i], bins=bin_edges, range=(0.0, 1.0), density=False)
                    hist = hist.astype(np.float64)
                    hist /= max(hist.sum(), 1.0)
                    pred_dense[i] = hist
                pred_full[test_mask] = pred_dense
            pred_full = normalize_rows_nonnegative(pred_full)
            logging.getLogger("bifurcation").info("[stage] render predicted density heatmap (ts mode) ...")
            plot_density_heatmap(
                r_values=r_values,
                bin_edges=bin_edges,
                densities=pred_full,
                title="Predicted density heatmap via TS surrogate",
                out_path="density_pred.png",
            )
            y_true_eval = densities_true[test_mask]
            y_pred_eval = pred_full[test_mask]
            mse, mean_js = evaluate_predictions(y_true=y_true_eval, y_pred=y_pred_eval)
            logging.getLogger("bifurcation").info(f"[ts] Test MSE: {mse:.6e}\n[ts] Test mean-JS: {mean_js:.6e}")
            # 额外指标与CSV
            if args.eval_extra:
                # 构造真值轨道（与 kept_ts 对齐的测试 r）
                true_trajs = []
                for r in r_values[test_mask]:
                    xs = simulate_logistic_series(r, num_iterations=cfg.num_iterations, num_discard=cfg.num_discard, x0=0.314)
                    true_trajs.append(xs[-kept_ts.shape[1]:] if xs.shape[0] >= kept_ts.shape[1] else np.pad(xs, (kept_ts.shape[1]-xs.shape[0],0), mode="edge"))
                true_trajs = np.stack(true_trajs, axis=0) if true_trajs else np.zeros_like(kept_ts)
                extra = evaluate_additional_metrics(
                    densities_true=y_true_eval,
                    densities_pred=y_pred_eval,
                    bin_edges=bin_edges,
                    traj_true=true_trajs,
                    traj_pred=kept_ts,
                    period_max=args.period_max,
                    period_tol=args.period_tol,
                )
                logging.getLogger("bifurcation").info(f"[extra] SSIM={extra['ssim']:.6e}, EMD={extra['emd']:.6e}, PeriodF1={extra['period_f1']:.6e}")
                # 保存CSV
                try:
                    with open("metrics_extra.csv", "w") as f:
                        f.write("metric,value\n")
                        for k,v in extra.items():
                            f.write(f"{k},{v}\n")
                except Exception:
                    pass
            if cfg.per_r_csv:
                try:
                    export_per_r_bucket_csv(
                        r_vals=r_values[test_mask],
                        dens_true=y_true_eval,
                        dens_pred=y_pred_eval,
                        bin_edges=bin_edges,
                        bucket_bins=cfg.per_r_bucket_bins,
                        out_csv="per_r_metrics.csv",
                    )
                except Exception:
                    pass
        # 自动双向：同参跑另一侧并导出LaTeX表
        if cfg.run_both:
            other = "dense" if cfg.train_side == "sparse" else "sparse"
            logging.getLogger("bifurcation").info(f"[both] running other side: {other}")
            cfg_other = cfg
            cfg_other.train_side = other
            # 简单复跑：递归调用 main 不便，这里以最少重复代码再跑一次核心评估（仅指标，不再出图）
            if other == "sparse":
                train_mask2 = r_values <= cfg.split_r
                test_mask2 = ~train_mask2
            else:
                train_mask2 = r_values > cfg.split_r
                test_mask2 = ~train_mask2
            X_tr2, Y_tr2 = build_ts_dataset(
                r_values=r_values[train_mask2], num_iterations=cfg.num_iterations, num_discard=cfg.num_discard,
                seed=cfg.seed+33, window=cfg.ts_window, horizon=tr_h, max_windows_per_r=cfg.ts_max_windows_per_r, progress=False)
            X_te2, Y_te2 = build_ts_dataset(
                r_values=r_values[test_mask2], num_iterations=cfg.num_iterations, num_discard=cfg.num_discard,
                seed=cfg.seed+34, window=cfg.ts_window, horizon=te_h, max_windows_per_r=min(cfg.ts_max_windows_per_r,400), progress=False)
            if cfg.model_type in ("mlp","ridge","svr"):
                ts_model2 = train_regressor(X_tr2, Y_tr2, cfg.model_type, cfg.hidden_layers, cfg.max_train_iter, cfg.learning_rate_init, cfg.seed+55, False)
            else:
                ModelCls = CurriculumGRUMLPRegressor if cfg.use_curriculum else GRUMLPRegressor
                ts_model2 = ModelCls(
                    window=cfg.ts_window, horizon=tr_h, hidden_size=cfg.gru_hidden_size, mlp_hidden=cfg.hidden_layers,
                    num_layers=cfg.gru_num_layers, dropout=cfg.gru_dropout, lr=cfg.learning_rate_init, batch_size=cfg.gru_batch_size,
                    max_epochs=max(1,min(cfg.max_train_iter,400)), seed=cfg.seed+56, residual=cfg.gru_residual, residual_lambda=cfg.gru_residual_lambda,
                    poly_degree=cfg.gru_poly_degree, input_jitter=cfg.ts_input_jitter, grad_clip=cfg.grad_clip, r_min=cfg.r_min, r_max=cfg.r_max,
                    r_poly_degree=cfg.gru_r_poly_degree, r_sin_cos=cfg.gru_r_sin_cos, r_norm_to_unit=cfg.r_norm_to_unit, weight_decay=cfg.gru_weight_decay,
                    rollout_k=cfg.rollout_consistency_k, rollout_lambda=cfg.rollout_consistency_lambda, mlp_dropout=cfg.mlp_dropout, spectral_norm=cfg.mlp_spectral_norm)
                if cfg.use_curriculum:
                    ts_model2.fit_with_curriculum(X_tr2, Y_tr2, verbose=False)
                else:
                    ts_model2.fit(X_tr2, Y_tr2, verbose=False)
            kept_ts2 = simulate_with_ts_surrogate_iterative(
                model=ts_model2, r_values=r_values[test_mask2], num_iterations=cfg.num_iterations, num_discard=cfg.num_discard,
                window=cfg.ts_window, seed=cfg.seed+57, progress=False)
            # 密度对齐
            pred_full2 = densities_true.copy()
            if kept_ts2.size > 0:
                pred_dense2 = np.zeros_like(densities_true[test_mask2])
                for i in range(kept_ts2.shape[0]):
                    hist,_ = np.histogram(kept_ts2[i], bins=bin_edges, range=(0.0,1.0), density=False)
                    hist = hist.astype(np.float64); hist /= max(hist.sum(), 1.0)
                    pred_dense2[i] = hist
                pred_full2[test_mask2] = pred_dense2
            pred_full2 = normalize_rows_nonnegative(pred_full2)
            y_true_eval2 = densities_true[test_mask2]; y_pred_eval2 = pred_full2[test_mask2]
            mse2, js2 = evaluate_predictions(y_true_eval2, y_pred_eval2)
            extra2 = {"ssim": float("nan"), "emd": float("nan"), "period_f1": float("nan")}
            try:
                # 构造真值轨道
                true_trajs2 = []
                for r in r_values[test_mask2]:
                    xs = simulate_logistic_series(r, num_iterations=cfg.num_iterations, num_discard=cfg.num_discard, x0=0.314)
                    true_trajs2.append(xs[-kept_ts2.shape[1]:] if xs.shape[0] >= kept_ts2.shape[1] else np.pad(xs,(kept_ts2.shape[1]-xs.shape[0],0),mode="edge"))
                true_trajs2 = np.stack(true_trajs2, axis=0) if true_trajs2 else np.zeros_like(kept_ts2)
                extra2 = evaluate_additional_metrics(y_true_eval2, y_pred_eval2, bin_edges, true_trajs2, kept_ts2, period_max=args.period_max, period_tol=args.period_tol)
            except Exception:
                pass
            # 导出LaTeX表（当前侧 + 另一侧）
            try:
                with open("results_table.tex","w") as f:
                    f.write("\\begin{tabular}{lcccc}\\hline\n")
                    f.write("side & MSE $\\downarrow$ & JS $\\downarrow$ & SSIM $\\uparrow$ & EMD $\\downarrow$ \\\\\\hline\n")
                    f.write(f"{cfg.train_side} & {mse:.2e} & {mean_js:.2e} & {extra['ssim']:.2e} & {extra['emd']:.2e} \\\\\n")
                    f.write(f"{other} & {mse2:.2e} & {js2:.2e} & {extra2['ssim']:.2e} & {extra2['emd']:.2e} \\\\\\hline\n")
                    f.write("\\end{tabular}\n")
            except Exception:
                pass

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()


