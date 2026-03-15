export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python main.py \
    --root '../OOD/ChChMiner_scaffold/' \
    --train_root '../OOD/ChChMiner_scaffold/train/' \
    --valid_root '../OOD/ChChMiner_scaffold/valid/' \
    --test_root '../OOD/ChChMiner_scaffold/test/' \
    --devices '3' \
    --filename "ft_ChChMiner_scaffold" \
    --opt_model './modelscope/galactica-1.3b' \
    --mode ft \
    --tune_gnn \
    --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' \
    --inference_batch_size 4 \
    --batch_size 4 \
    --max_len 30  \
    --NAS \
    --DDI \
    --transform 'Motif' \
    --llm_tune lora \
    --max_epochs 10 \
    --caption_eval_epoch 1 \
    --save_every_n_epochs 1 \
    --num_beams 1 \
    --use_att \
    --stage2_path "all_checkpoints/pretrain_ChChMiner_scaffold/last.ckpt"

# python main.py \
#     --root '../OOD/Drugbank_scaffold/' \
#     --train_root '../OOD/Drugbank_scaffold/train/' \
#     --valid_root '../OOD/Drugbank_scaffold/valid/' \
#     --test_root '../OOD/Drugbank_scaffold/test/' \
#     --devices '3' \
#     --filename "ft_Drugbank_scaffold" \
#     --opt_model './modelscope/galactica-1.3b' \
#     --mode ft \
#     --tune_gnn \
#     --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' \
#     --inference_batch_size 4 \
#     --batch_size 4 \
#     --max_len 30  \
#     --NAS \
#     --drugbank \
#     --transform 'Motif' \
#     --llm_tune lora \
#     --max_epochs 10 \
#     --caption_eval_epoch 1 \
#     --save_every_n_epochs 1 \
#     --num_beams 1 \
#     --use_att \
#     --stage2_path "all_checkpoints/pretrain_Drugbank_scaffold/last.ckpt"