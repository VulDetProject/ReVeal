export PYTHONPATH='../../Vuld_SySe/':$PYTHONPATH
for i in 1 2 3 4 5; do
	python ../../Vuld_SySe/vul_det_main.py \
		--train_file ../../data/SySeVR/${2}-devign.json\
		--model_path ../../models/sys_${2}_train_on_devign.bin \
		--word_to_vec ../../data/Word2Vec/li_et_al_wv \
		--batch_size $1 \
		--model_type bigru >> ../../outputs/sys_${2}_train_on_devign.log
done
