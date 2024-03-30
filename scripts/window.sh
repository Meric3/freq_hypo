# python ../src/train.py --config ../config/window/10.yml --features none --model 1d_cnn --gpu 1 --exp time_window_10_
# python ../src/train.py --config ../config/window/10_f.yml --features all --model comb --gpu 1 --exp time_window_10_f
# python ../src/train.py --config ../config/window/20.yml --features none --model 1d_cnn --gpu 1 --exp time_window_20_
# python ../src/train.py --config ../config/window/20_f.yml --features all --model comb --gpu 1 --exp time_window_20_f
python ../src/train.py --config ../config/window/40.yml --features none --model 1d_cnn --gpu 1 --exp time_window_40_
python ../src/train.py --config ../config/window/40_f.yml --features all --model comb --gpu 1 --exp time_window_40_f
python ../src/train.py --config ../config/window/50.yml --features none --model 1d_cnn --gpu 1 --exp time_window_50_
python ../src/train.py --config ../config/window/50_f.yml --features all --model comb --gpu 1 --exp time_window_50_f
python ../src/train.py --config ../config/window/60.yml --features none --model 1d_cnn --gpu 1 --exp time_window_60_
python ../src/train.py --config ../config/window/60_f.yml --features all --model comb --gpu 1 --exp time_window_60_f
# python ../train.py --config ../config/0916_one_time.yml --features ecg_time --gpu 1 --exp time_ecg
# python ../train.py --config ../config/0916_one_time.yml --features co2_time --gpu 1 --exp time_co2
