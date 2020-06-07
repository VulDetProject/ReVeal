export PYTHONPATH='../Vuld_SySe/':$PYTHONPATH
for i in 1 2 3 4 5; do
	python ../Vuld_SySe/draper_main.py \
		--train_file ../data/draper/devign.json \
		--model_path ../models/draper/train_devign.bin >> ../outputs/draper_train_devign.txt;
done
