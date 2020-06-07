dataset=$1;
batch_size=$2;
export PYTHONPATH='../Vuld_SySe/':$PYTHONPATH
for i in 1 2 3 4 5 ; do
	python ../Vuld_SySe/draper_main.py \
		--train_file ../data/draper/${dataset}.json \
		--model_path ../models/transformers/${dataset}_intra_dataset-${i}.bin  \
		--batch_size ${batch_size} \
		--intra_dataset >> ../outputs/${dataset}-Transformer-intra.log;
done

