## Choice of dataset: devign, chrome_debian
## batch_size 128

dataset=$1;
batch_size=$2;
export PYTHONPATH='../Vuld_SySe/':$PYTHONPATH
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
	python ../Vuld_SySe/draper_main.py \
		--train_file ../data/full_experiment_real_data_processed/${dataset}-draper.json \
		--model_path ../models/draper/${dataset}_intra_dataset-${i}-new_data.bin  \
		--batch_size ${batch_size} \
		--intra_dataset >> ../outputs/${dataset}-Draper-new_intra_dataset.log;
done
