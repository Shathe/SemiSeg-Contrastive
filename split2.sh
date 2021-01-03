echo "deeplabv2 split 2"
python3 trainSSL.py --config ./configs/configSSL_city_1_30_split2.json --name SSL --gpus 1

echo "deeplabv3+ split 2"
python3 trainSSL_fullres.py --config ./configs/configSSL_city_1_30_split2_v3.json --name SSL --gpus 1
