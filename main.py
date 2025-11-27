"""
Logistic Map Bifurcation Prediction
"""
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import ExperimentConfig
from models import (
    generate_bifurcation_data,
    build_density_matrix,
    compute_lyapunov_for_r_values,
    train_regressor,
    GRUMLPRegressor
)
from data_utils import (
    symmetrize_densities,
    smooth_along_r,
    normalize_rows_nonnegative,
    sample_scatter_from_densities,
    build_map_training_data,
    build_ts_dataset
)
from visualization import (
    plot_bifurcation_scatter,
    plot_density_heatmap,
    plot_lyapunov_curve,
    plot_bifurcation_scatter_pred
)
from evaluation import evaluate_predictions, evaluate_additional_metrics


def setup_logging(log_file: str = None, level: str = "INFO") -> logging.Logger:
    """Configure logging"""
    logger = logging.getLogger("bifurcation")
    if logger.handlers:
        logger.setLevel(level.upper())
        return logger
    
    logger.setLevel(level.upper())
    fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    
    return logger


def parse_args() -> ExperimentConfig:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Predict logistic map bifurcation behavior using neural networks"
    )
    
    # Basic parameters
    parser.add_argument("--r_min", type=float, default=2.5)
    parser.add_argument("--r_max", type=float, default=4.0)
    parser.add_argument("--num_r", type=int, default=400)
    parser.add_argument("--split_r", type=float, default=3.46994)
    parser.add_argument("--seed", type=int, default=42)
    
    # Model parameters
    parser.add_argument("--hidden", type=str, default="256,256")
    parser.add_argument("--model", type=str, choices=["mlp", "ridge", "gru"], default="mlp")
    parser.add_argument("--max_train_iter", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    # Running modes
    parser.add_argument("--mode", type=str, choices=["map", "density", "ts"], default="map")
    parser.add_argument("--train_side", type=str, choices=["sparse", "dense"], default="sparse")
    
    # Time series parameters
    parser.add_argument("--ts_window", type=int, default=16)
    parser.add_argument("--ts_horizon", type=int, default=16)
    parser.add_argument("--ts_strategy", type=str, choices=["direct", "iter"], default="iter")
    
    # GRU parameters
    parser.add_argument("--gru_hidden_size", type=int, default=256)
    parser.add_argument("--gru_num_layers", type=int, default=1)
    parser.add_argument("--gru_batch_size", type=int, default=256)
    
    # Others
    parser.add_argument("--no_progress", dest="progress", action="store_false")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_file", type=str, default=None)
    
    args = parser.parse_args()
    
    # Parse hidden layers
    hidden_layers = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    
    # Create configuration
    cfg = ExperimentConfig(
        r_min=args.r_min,
        r_max=args.r_max,
        num_r=args.num_r,
        split_r=args.split_r,
        seed=args.seed,
        hidden_layers=hidden_layers,
        model_type=args.model,
        max_train_iter=args.max_train_iter,
        learning_rate_init=args.lr,
        mode=args.mode,
        train_side=args.train_side,
        ts_window=args.ts_window,
        ts_horizon=args.ts_horizon,
        ts_strategy=args.ts_strategy,
        gru_hidden_size=args.gru_hidden_size,
        gru_num_layers=args.gru_num_layers,
        gru_batch_size=args.gru_batch_size,
        progress=args.progress,
        mlp_verbose=args.verbose,
        log_file=args.log_file,
    )
    
    return cfg


def run_map_mode(cfg: ExperimentConfig, r_values: np.ndarray, 
                densities_true: np.ndarray, bin_edges: np.ndarray,
                train_mask: np.ndarray, test_mask: np.ndarray, logger: logging.Logger):
    """Run map mode"""
    from simulation import simulate_with_map_surrogate
    
    logger.info("Building map training data...")
    X_pairs, y_pairs = build_map_training_data(
        r_values=r_values[train_mask],
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        seed=cfg.seed,
        max_pairs_per_r=cfg.map_pairs_per_r,
        progress=cfg.progress,
    )
    
    logger.info(f"Training map model, pairs={X_pairs.shape[0]:,}...")
    map_model = train_regressor(
        X_train=X_pairs,
        y_train=y_pairs,
        model_type="mlp",
        hidden_layers=cfg.hidden_layers,
        max_iter=cfg.max_train_iter,
        learning_rate_init=cfg.learning_rate_init,
        seed=cfg.seed,
        verbose=cfg.mlp_verbose,
    )
    
    logger.info("Simulating trajectories...")
    kept = simulate_with_map_surrogate(
        model=map_model,
        r_values=r_values[test_mask],
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        seed=cfg.seed + 7,
        progress=cfg.progress,
    )
    
    # Generate prediction scatter points
    T_keep = kept.shape[1]
    take = min(cfg.scatter_per_r, T_keep)
    if take > 0:
        sel = np.arange(T_keep - take, T_keep)
        x_pred = kept[:, sel].reshape(-1)
        r_rep = np.repeat(r_values[test_mask], take)
    else:
        x_pred, r_rep = np.array([]), np.array([])
    
    # Training region scatter points
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
    
    # Build density matrix
    pred_full = densities_true.copy()
    if kept.size > 0:
        pred_dense = np.zeros_like(densities_true[test_mask])
        for i in range(kept.shape[0]):
            hist, _ = np.histogram(kept[i], bins=bin_edges, range=(0.0, 1.0))
            hist = hist.astype(np.float64)
            hist /= max(hist.sum(), 1.0)
            pred_dense[i] = hist
        pred_full[test_mask] = pred_dense
    
    pred_full = normalize_rows_nonnegative(pred_full)
    
    plot_density_heatmap(
        r_values=r_values,
        bin_edges=bin_edges,
        densities=pred_full,
        title="Predicted density (map mode)",
        out_path="density_pred.png",
    )
    
    # Evaluation
    y_true_eval = densities_true[test_mask]
    y_pred_eval = pred_full[test_mask]
    mse, mean_js = evaluate_predictions(y_true_eval, y_pred_eval)
    logger.info(f"Test MSE: {mse:.6e}, Mean JS: {mean_js:.6e}")


def run_ts_mode(cfg: ExperimentConfig, r_values: np.ndarray,
               densities_true: np.ndarray, bin_edges: np.ndarray,
               train_mask: np.ndarray, test_mask: np.ndarray, logger: logging.Logger):
    """Run time series mode"""
    from simulation import simulate_with_ts_surrogate
    
    logger.info("Building time-series dataset...")
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
    
    logger.info(f"Training TS model ({cfg.model_type}), samples={X_tr.shape[0]:,}...")
    if cfg.model_type == "gru":
        ts_model = GRUMLPRegressor(
            window=cfg.ts_window,
            horizon=tr_h,
            hidden_size=cfg.gru_hidden_size,
            mlp_hidden=cfg.hidden_layers,
            num_layers=cfg.gru_num_layers,
            batch_size=cfg.gru_batch_size,
            lr=cfg.learning_rate_init,
            max_epochs=min(cfg.max_train_iter, 400),
            seed=cfg.seed,
            residual=cfg.gru_residual,
        )
    else:
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
    
    ts_model.fit(X_tr, Y_tr, verbose=cfg.mlp_verbose)
    
    if cfg.ts_strategy == "iter":
        logger.info("Simulating trajectories via TS surrogate...")
        kept_ts = simulate_with_ts_surrogate(
            model=ts_model,
            r_values=r_values[test_mask],
            num_iterations=cfg.num_iterations,
            num_discard=cfg.num_discard,
            window=cfg.ts_window,
            seed=cfg.seed + 7,
            progress=cfg.progress,
        )
        
        # Evaluation and visualization (similar to map mode)
        pred_full = densities_true.copy()
        if kept_ts.size > 0:
            pred_dense = np.zeros_like(densities_true[test_mask])
            for i in range(kept_ts.shape[0]):
                hist, _ = np.histogram(kept_ts[i], bins=bin_edges, range=(0.0, 1.0))
                hist = hist.astype(np.float64)
                hist /= max(hist.sum(), 1.0)
                pred_dense[i] = hist
            pred_full[test_mask] = pred_dense
        
        pred_full = normalize_rows_nonnegative(pred_full)
        
        y_true_eval = densities_true[test_mask]
        y_pred_eval = pred_full[test_mask]
        mse, mean_js = evaluate_predictions(y_true_eval, y_pred_eval)
        logger.info(f"Test MSE: {mse:.6e}, Mean JS: {mean_js:.6e}")
        
        plot_density_heatmap(
            r_values=r_values,
            bin_edges=bin_edges,
            densities=pred_full,
            title="Predicted density (TS mode)",
            out_path="density_pred.png",
        )


def main():
    """Main function"""
    cfg = parse_args()
    logger = setup_logging(cfg.log_file, cfg.log_level)
    
    logger.info("Starting bifurcation prediction experiment...")
    logger.info(f"Mode: {cfg.mode}, Model: {cfg.model_type}, Train side: {cfg.train_side}")
    
    # Generate r values
    r_values = np.linspace(cfg.r_min, cfg.r_max, cfg.num_r)
    
    # 1. Generate true bifurcation data
    logger.info("Generating true bifurcation data...")
    r_points, x_points = generate_bifurcation_data(
        r_values=r_values,
        num_iterations=cfg.num_iterations,
        num_discard=cfg.num_discard,
        seed=cfg.seed,
        progress=cfg.progress,
    )
    plot_bifurcation_scatter(r_points, x_points, cfg.split_r, "bifurcation_scatter.png")
    
    # 2. Build true density matrix
    logger.info("Building true density matrix...")
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
    if cfg.smooth_sigma > 0:
        densities_true = smooth_along_r(densities_true, cfg.smooth_sigma)
    
    plot_density_heatmap(
        r_values, bin_edges, densities_true,
        "True density heatmap", "density_true.png"
    )
    
    # 3. Compute Lyapunov exponents
    if cfg.lyapunov:
        logger.info("Computing Lyapunov exponents...")
        lyap = compute_lyapunov_for_r_values(
            r_values, cfg.num_iterations, cfg.num_discard,
            cfg.seed, cfg.progress
        )
        plot_lyapunov_curve(r_values, lyap, cfg.split_r, "lyapunov_true.png")
    
    # 4. Split training/test regions
    if cfg.train_side == "sparse":
        train_mask = r_values <= cfg.split_r
    else:
        train_mask = r_values > cfg.split_r
    test_mask = ~train_mask
    
    logger.info(f"Train samples: {train_mask.sum()}, Test samples: {test_mask.sum()}")
    
    # 5. Run experiment according to mode
    if cfg.mode == "map":
        run_map_mode(cfg, r_values, densities_true, bin_edges, 
                    train_mask, test_mask, logger)
    elif cfg.mode == "ts":
        run_ts_mode(cfg, r_values, densities_true, bin_edges,
                   train_mask, test_mask, logger)
    else:
        logger.error(f"Unsupported mode: {cfg.mode}")
    
    logger.info("Experiment completed!")
    plt.show()


if __name__ == "__main__":
    main()
