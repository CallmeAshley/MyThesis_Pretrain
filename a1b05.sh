export CUDA_VISIBLE_DEVICES="0,1,2,3"


torchrun --standalone --nnodes=1 --nproc_per_node=4 /mai_nas/LSH/SparK/pretrain/main.py


