export PYTHONPATH='../../Vuld_SySe/':$PYTHONPATH
python ../../Vuld_SySe/vul_det_main.py \
	--train_file  ../../data/SySeVR/Array_usage-processed.json\
	--test_file  ../../data/SySeVR/Array_usage-chrome_debian.json\
	--model_path ../../models/sys_array_test_on_chrome_deb.bin \
	--word_to_vec ../../data/Word2Vec/li_et_al_wv \
	--batch_size $1 --test_only \
	--model_type bigru > ../../outputs/sys_array_test_on_chrome_deb.log
