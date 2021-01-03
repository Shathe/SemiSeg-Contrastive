echo "deeplabv2 split 0"
python3 trainSSL.py --config ./configs/configSSL_city_1_30_split0.json --name SSL --gpus 1

echo "deeplabv3+ split 0"
python3 trainSSL_fullres.py --config ./configs/configSSL_city_1_30_split0_v3.json --name SSL --gpus 1
