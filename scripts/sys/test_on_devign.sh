export PYTHONPATH='../../Vuld_SySe/':$PYTHONPATH
python ../../Vuld_SySe/vul_det_main.py \
	--train_file  ../../data/SySeVR/${2}-processed.json\
	--test_file  ../../data/SySeVR/${2}-devign.json\
	--model_path ../../models/sys_${2}_test_on_devign.bin \
	--word_to_vec ../../data/Word2Vec/li_et_al_wv \
	--batch_size $1 --test_only \
	--model_type bigru >  ../../outputs/sys_${2}_test_on_devign.log
