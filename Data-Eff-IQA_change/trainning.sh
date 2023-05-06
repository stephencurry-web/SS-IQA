### LIVEC
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49958  main.py \
--cfg configs/Pure/vit_small_pre_coder_livec.yaml \
--data-path /data/qgy/IQA-Dataset/ChallengeDB_release \
--output log \
--tensorboard \
--tag vit_warm_decoder_j6_224_livec_posbiHW \
--repeat \
--rnum 10

### KONIQ-10K
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49958  main.py \
--cfg configs/Pure/vit_small_pre_coder_koniq.yaml \
--data-path /data/qgy/IQA-Dataset/koinq-10k \
--output log \
--tensorboard \
--tag vit_warm_decoder_j6_224_livec_posbiHW \
--repeat \
--rnum 10

### LIVE
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49958  main.py \
--cfg configs/Pure/vit_small_pre_coder_live.yaml \
--data-path /data/qgy/IQA-Dataset/live/databaserelease2 \
--output log \
--tensorboard \
--tag vit_warm_decoder_j6_224_live_posbiHW \
--repeat \
--rnum 10

### TID2013
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49958  main.py \
--cfg configs/Pure/vit_small_pre_coder_tid.yaml \
--data-path /data/qgy/IQA-Dataset/tid2013 \
--output log \
--tensorboard \
--tag vit_warm_decoder_j6_224_tid_posbiHW \
--repeat \
--rnum 10

### CSIQ
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49958  main.py \
--cfg configs/Pure/vit_small_pre_coder_csiq.yaml \
--data-path /data/qgy/IQA-Dataset/CSIQ \
--output log \
--tensorboard \
--tag vit_warm_decoder_j6_224_csiq_posbiHW \
--repeat \
--rnum 10

###KADID
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49958  main.py \
--cfg configs/Pure/vit_small_pre_coder_kadid.yaml \
--data-path /data/qgy/IQA-Dataset/kadid/kadid10k \
--output log \
--tensorboard \
--tag vit_warm_decoder_j6_224_kadid_posbiHW \
--repeat \
--rnum 10

###SPAQ
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49958  main.py \
--cfg configs/Pure/vit_small_pre_coder_spaq.yaml \
--data-path /data/qgy/IQA-Dataset/SPAQ \
--output log \
--tensorboard \
--tag vit_warm_decoder_j6_224_spaq_posbiHW \
--repeat \
--rnum 10

###LIVEFB
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=6 torchrun --nnodes 1 --nproc_per_node 4 --master_port 49958  main.py \
--cfg configs/Pure/vit_small_pre_coder_livefb.yaml \
--data-path /data/qgy/IQA-Dataset/liveFB \
--output log \
--tensorboard \
--tag vit_warm_decoder_j6_224_livefb_posbiHW \
--repeat \
--rnum 10