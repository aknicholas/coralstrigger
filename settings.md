python -m snr_scan --fit-json plots/threshold_model_400_beams.json --target-global-rate 10 --n-beams 966 --angle-scan --phi0 90 --theta0 0 --span-phi 23 --span-theta 23 --ang-res 1 --target-eff 0.5 --window 160 --step 40

python power_cuts.py --clean --quadrant-only --half-span 31 --step 1 \
  --beam-span-phi 3 --beam-span-theta 3 --beam-res 0.25 \
  --radius 30 --force --contour-db -1
  
python diagnostics.py thresholds --power-file noise/power_160_40.npy --noise-file noise/simulated_noise.npy --window 160 --step 40

