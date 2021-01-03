echo "deeplabv2 split 1"
python3 trainSSL.py --config ./configs/configSSL_city_1_30_split1.json --name SSL --gpus 1

echo "deeplabv3+ split 1"
python3 trainSSL_fullres.py --config ./configs/configSSL_city_1_30_split1_v3.json --name SSL --gpus 1
