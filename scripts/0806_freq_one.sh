
#0418
python ../train.py --config ../config/one_freq.yml --features ecg_freq --gpu 1 --exp freq_ecg
python ../train.py --config ../config/one_freq.yml --features co2_freq --gpu 1 --exp freq_co2
python ../train.py --config ../config/one_freq.yml --features ple_freq --gpu 1 --exp freq_ple
python ../train.py --config ../config/one_freq.yml --features abp_freq --gpu 1 --exp freq_abp
