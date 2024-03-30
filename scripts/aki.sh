python ../src/train.py --config ../config/aki/time.yml --features none --model 1d_cnn --gpu 0 --exp aki_time
python ../src/train.py --config ../config/aki/freq.yml --features all --model comb --gpu 0 --exp aki_freq
# python ../train.py --config ../config/0916_one_time.yml --features co2_time --gpu 1 --exp time_co2
