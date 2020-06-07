## Choice of dataset: devign, chrome_debian
## batch_size 128
## model_type bigru

dataset=$1;
batch_size=$2;
model_type=$3;
export PYTHONPATH='../Vuld_SySe/':$PYTHONPATH
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
	python ../Vuld_SySe/vul_det_main.py \
		--train_file ../data/full_experiment_real_data_processed/${dataset}-syse.json \
		--model_path ../models/SySeVR/${dataset}_new_intra_dataset-${i}.bin \
		--word_to_vec ../data/Word2Vec/li_et_al_wv \
		--batch_size ${batch_size} \
		--model_type ${model_type} --intra_dataset >> ../outputs/${dataset}-SySeVR-${model_type}-new_intra_dataset.log;
done
