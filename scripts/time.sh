# python ../train.py --config ../config/0916_time.yml --features none --model 1d_cnn --gpu 1 --exp time_multi
python ../train.py --config ../config/0916_one_time.yml --features ecg_time --gpu 1 --exp time_ecg
python ../train.py --config ../config/0916_one_time.yml --features co2_time --gpu 1 --exp time_co2