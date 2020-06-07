echo "CHROME AND DEBIAN DATA"
echo "==================================================================================="
python create_ggnn_data.py;
echo "==================================================================================="
echo "DEVIGN DATA"
python create_ggnn_data.py --project devign --csv ../data/neurips_parsed/parsed_results/ --src ../data/neurips_parsed/neurips_data/ --wv ../data/neurips_parsed/raw_code_neurips.100 --output ../data/full_experiment_real_data/devign/devign.json
