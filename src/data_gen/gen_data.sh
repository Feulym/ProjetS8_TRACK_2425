#!/bin/bash

durations=("120 300")
sigma_loi=(0.01 0.1)
tau=(60 40 5)
sigma_m2=(0.03849 8.898 24.06)
noised=1 # 1 if True, 0 if False

data_dir="data"
mkdir -p "$data_dir"

for duration in "${durations[@]}"; do
    IFS=' ' read -r min_N max_N <<< "$duration"
    for sigma in "${sigma_loi[@]}"; do
        for i in "${!tau[@]}"; do
            t="${tau[$i]}"
            sm2="${sigma_m2[$i]}"
            output_file="$data_dir/data_min${min_N}_max${max_N}_sigma${sigma}_tau${t}_sigmaM${sm2}.h5"

            python3 data_gen.py \
                --output_file "$output_file" \
                --n_samples 1 \
                --min_N "$min_N" \
                --max_N "$max_N" \
                --Tech 1.0 \
                --min_sigma2 "$sigma" \
                --max_sigma2 "$sigma" \
                --min_tau "$t" \
                --max_tau "$t" \
                --min_sigma_m2 "$sm2" \
                --max_sigma_m2 "$sm2" \
                --noised "$noised"
        done
    done
done
