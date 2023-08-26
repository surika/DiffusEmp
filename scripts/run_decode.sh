CUDA_VISIBLE_DEVICES=0 python -u run_decode.py \
--model_dir /path-to-repo/diffusemp/diffusion_models/diffusemp_ed_h128_lr0.0001_t2000_sqrt_lossaware_seed102_mancity_mask-fine-vm \
--seed 123 \
--split test
# --pattern ema_0.9999_010000