export PYTHONPATH='../Vuld_SySe/':$PYTHONPATH
for i in 1 2 3 4 5; do
	python ../Vuld_SySe/draper_main.py \
		--train_file ../data/draper/chrome_debian.json \
		--model_path ../models/draper/train_chrome_debian.bin >> ../outputs/draper_train_chrome_debian.txt;
done
