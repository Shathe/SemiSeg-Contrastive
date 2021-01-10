echo "full coco pre-trained"
python3 trainSupervised.py --config ./configs/configSSL_city_1_1.json --name SSL --gpus 1
echo "full imagenet pre-trained"
python3 trainSupervised_imagenet.py --config ./configs/configSSL_city_1_1.json --name SSL --gpus 1