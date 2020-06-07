export PYTHONPATH='../Vuld_SySe/':$PYTHONPATH
python ../Vuld_SySe/draper_main.py \
	--train_file ../data/draper/juliet.json \
	--test_file ../data/draper/chrome_debian.json \
	--model_path ../models/draper/juliet_test_chrome_debian.bin  >> ../outputs/juliet_test_chrome_debian.txt
