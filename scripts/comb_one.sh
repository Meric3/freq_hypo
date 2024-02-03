# 0418
python ../train.py --features ecg_freq --model comb --config ../config/one_freq.yml --gpu 1 --exp freq_ecg
python ../train.py --features co2_freq --model comb --config ../config/one_freq.yml --gpu 1 --exp freq_co2
