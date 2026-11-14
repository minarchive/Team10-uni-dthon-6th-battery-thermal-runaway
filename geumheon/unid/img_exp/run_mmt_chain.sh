#!/bin/bash
# effb0 fold 5개가 끝나면 → noimg fold 5개 병렬 → ResNet50 fold 5개 병렬 (각 3스레드), 단계마다 합치기
cd /home/nxcolelxu/unid/img_exp
source ../venv/bin/activate
run_stage () {   # $1 = img, $2 = out
  echo "[$(date +%T)] start $2"
  for k in 0 1 2 3 4; do
    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 python -c 'import torch,sys,runpy; torch.set_num_threads(3); sys.argv=["mm_transformer.py"]+sys.argv[1:]; runpy.run_path("mm_transformer.py", run_name="__main__")' \
      --data_dir ../data/battery_data_light --img $1 --seeds 0 --folds $k --out $2 > logs/${2}_f$k.log 2>&1 &
  done
  wait
  python mm_transformer.py --data_dir ../data/battery_data_light --img $1 --seeds 0 --out $2 --merge_only > logs/${2}_merge.log 2>&1
  echo "[$(date +%T)] done $2: $(grep 'CV post .* over' logs/${2}_merge.log)"
}
while pgrep -f "img effb0 --seeds 0 --folds" >/dev/null; do sleep 30; done
echo "[$(date +%T)] effb0 folds finished"
run_stage none mmt_noimg_cpu
run_stage resnet50 mmt_r50_cpu
echo "[$(date +%T)] all stages done"
