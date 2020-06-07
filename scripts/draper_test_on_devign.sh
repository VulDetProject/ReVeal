export PYTHONPATH='../Vuld_SySe/':$PYTHONPATH
python ../Vuld_SySe/draper_main.py \
	--train_file ../data/draper/train_sampled.json \
	--test_file ../data/draper/devign.json \
	--model_path ../models/draper_test_devign.bin --test_only > ../outputs/draper_test_devign.txt
