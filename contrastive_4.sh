echo "Wise"
python3 contrastive_wise_manual.py --config ./configs/configSSL_city_1_30_split0.json --name SSL --gpus 1
echo "Use all"
python3 contrastive_all.py --config ./configs/configSSL_city_1_30_split0.json --name SSL --gpus 1


