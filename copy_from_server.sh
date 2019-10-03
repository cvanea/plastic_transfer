#!/usr/bin/env bash

for value in {50..56}
do
    gsutil cp -r gs://plastic_transfer_bucket/results/exp_15/run_$value results_cloud/results/exp_15
done
