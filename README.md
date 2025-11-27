# Logistic-Map-Pred
python gru_mlp_bifurcation.py \
  --mode ts --model gru --ts_strategy iter \
  --r_min 2.5 --r_max 4.0 --num_r 400 \
  --split_r 3.56995 --train_side sparse \
  --ts_window 64 --ts_horizon 1 \
  --gru_poly_degree 2 --gru_residual --gru_residual_lambda 1e-3 \
  --rollout_consistency_k 4 --rollout_consistency_lambda 1e-3 \
  --mlp_spectral_norm --mlp_dropout 0.1 \
  --use_curriculum --curriculum_stages 3 \

双向实验
# 混沌->有序
--mode ts --model gru --train_side sparse
# 有序->混沌
--mode ts --model gru --train_side dense
