echo "deeplabv2 split 1   1/8"
python3 trainSSL2.py --config ./configs/configSSL_city_1_8_split1.json --name SSL --gpus 1
echo "deeplabv2 split 1   1/4"
python3 trainSSL2.py --config ./configs/configSSL_city_1_4_split1.json --name SSL --gpus 1
