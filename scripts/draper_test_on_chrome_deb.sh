export PYTHONPATH='../Vuld_SySe/':$PYTHONPATH
python ../Vuld_SySe/draper_main.py \
	--train_file ../data/draper/train_sampled.json \
	--test_file ../data/draper/chrome_debian.json \
	--model_path ../models/draper/test_chrome_debian.bin --test_only >> ../outputs/draper_test_chrome_debian.txt
