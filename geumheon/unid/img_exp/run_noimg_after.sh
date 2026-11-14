#!/bin/bash
# effb0 fold 프로세스 5개가 모두 끝나면 noimg를 fold 병렬(5 × 2스레드)로 실행
cd /home/nxcolelxu/unid/img_exp
source ../venv/bin/activate
while pgrep -f "img effb0 --seeds 0 --folds" >/dev/null; do sleep 30; done
echo "effb0 done at $(date), starting noimg folds"
for k in 0 1 2 3 4; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -c 'import torch,sys,runpy; torch.set_num_threads(2); sys.argv=["mm_transformer.py"]+sys.argv[1:]; runpy.run_path("mm_transformer.py", run_name="__main__")' \
    --data_dir ../data/battery_data_light --img none --seeds 0 --folds $k --out mmt_noimg_cpu > logs/mmt_noimg_cpu_f$k.log 2>&1 &
done
wait
python mm_transformer.py --data_dir ../data/battery_data_light --img none --seeds 0 --out mmt_noimg_cpu --merge_only > logs/mmt_noimg_cpu_merge.log 2>&1
echo "noimg done at $(date)"
