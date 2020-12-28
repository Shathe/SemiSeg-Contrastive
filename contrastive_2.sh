echo "Contrastive learn N class detach"
python3 contrastive_learned_per_class_detach.py --config ./configs/configSSL_city_1_30_split0.json --name SSL --gpus 1
echo "Contrastive learn N class NOdetach"
python3 contrastive_learned_per_class.py --config ./configs/configSSL_city_1_30_split0.json --name SSL --gpus 1

