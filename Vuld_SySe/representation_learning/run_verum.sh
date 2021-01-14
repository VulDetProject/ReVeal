#python api_test.py --dataset chrome_debian --features wo_ggnn;
#for l in 0 5 2 4 3; do
#  python api_test.py --dataset chrome_debian/imbalanced --features ggnn --num_layers $l;
#done

python api_test.py --dataset chrome_debian/imbalanced --features ggnn;