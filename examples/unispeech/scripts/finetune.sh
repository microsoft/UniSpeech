model_path=~/test
pretrained_model=/datablob/users/v-chengw/commonvoice_model/ptmtls2_en1350.large.lr1e-3.64gpu.fgm0.1/checkpoint_best.pt
train_subset=trainSeqs_1.0_uniform_new_version_16k
valid_subset=valSeqs_1.0_uniform_new_version_16k

mkdir -p ${model_path}
WORLD_SIZE=4
updata_freq=1


python train.py --distributed-world-size $WORLD_SIZE --distributed-port 0 /datablob/users/v-chengw/data/commonvoice_20200622/resource/nl --save-dir ${model_path} --post-process word --train-subset ${train_subset} --valid-subset ${valid_subset} --no-epoch-checkpoints --best-checkpoint-metric uer --num-workers 4 --max-update 20000 --sentence-avg --task audio_pretraining --arch wav2vec_ctc --w2v-path ${pretrained_model} --labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.75 --layerdrop 0.1 --mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.25 --zero-infinity --feature-grad-mult 0.0 --freeze-finetune-updates 2000 --validate-after-updates 2000 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 2000 --hold-steps 8000 --decay-steps 10000 --final-lr-scale 0.05 --activation-dropout 0.1  --dropout 0.1 --attention-dropout 0.1 --final-dropout 0.1 --dropout-input 0.1 --criterion ctc --max-tokens 1000000 --seed 1337 --log-format json --log-interval 100 --ddp-backend no_c10d --fp16 --update-freq ${updata_freq} --dict-path /datablob/users/v-chengw/data/commonvoice_20200622/common_voices_splits/nl/phonesMatches_reduced.json --save-interval 10 --validate-interval 10 --normalize
