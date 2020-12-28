echo "Contrastive learn one detach"
python3 contrastive_learned_one.py --config ./configs/configSSL_city_1_30_split0.json --name SSL --gpus 1
echo "Contrastive learn one NOdetach"
python3 contrastive_learned_one_detach.py --config ./configs/configSSL_city_1_30_split0.json --name SSL --gpus 1

