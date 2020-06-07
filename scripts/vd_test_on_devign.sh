export PYTHONPATH='../Vuld_SySe/':$PYTHONPATH
python ../Vuld_SySe/vul_det_main.py --train_file ../data/VulDeePecker/CWE-119-processed.json  ../data/VulDeePecker/CWE-399-processed.json --test_file ../data/VulDeePecker/devign.json --model_path ../models/vuld_test_on_devign.bin --word_to_vec ../data/Word2Vec/li_et_al_wv --batch_size $1 --test_only >> ../outputs/vuld_test_on_devign.log
