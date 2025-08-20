mkdir -p results
python settings_loader.py > results/intrinsic_calibration.log
python intrinsic.py >> results/intrinsic_calibration.log
python visualize_int.py >> results/intrinsic_calibration.log