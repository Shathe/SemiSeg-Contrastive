echo "No contrastive"
python3 base.py --config ./configs/configSSL_city_1_30_split0.json --name SSL --gpus 1
echo "Contrastive Random"
python3 contrastive_random.py --config ./configs/configSSL_city_1_30_split0.json --name SSL --gpus 1

