# python ../train.py --config ../config/0916_freq.yml --features all --model comb --gpu 0 --exp comb_multi
python ../train.py --config ../config/0916_one_freq.yml --features ecg_freq --model comb --gpu 0 --exp freq_ecg
python ../train.py --config ../config/0916_one_freq.yml --features co2_freq --model comb --gpu 0 --exp freq_co2