echo "deeplabv2 split 0   1/8"
python3 trainSSL.py --config ./configs/configSSL_city_1_8_split0.json --name SSL --gpus 1

echo "deeplabv2 split 0   1/4"
python3 trainSSL.py --config ./configs/configSSL_city_1_4_split0.json --name SSL --gpus 1
