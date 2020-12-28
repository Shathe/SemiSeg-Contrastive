
echo "1/8 dataset strong aug"
python3 trainSupervised2.py --config ./configs/configSSL_city_1_8.json --name SSL --gpus 1

echo "1/1 dataset strong aug"
python3 trainSupervised2.py --config ./configs/configSSL_city_1_1.json --name SSL --gpus 1
