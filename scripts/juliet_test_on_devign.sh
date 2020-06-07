export PYTHONPATH='../Vuld_SySe/':$PYTHONPATH
python ../Vuld_SySe/draper_main.py \
	--train_file ../data/draper/juliet.json \
	--test_file ../data/draper/devign.json \
	--model_path ../models/juliet_test_devign.bin --test_only> ../outputs/juliet_test_devign.txt
