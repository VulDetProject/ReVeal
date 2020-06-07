#python api_test.py --dataset devign --features wo_ggnn --lambda1 0 --lambda2 0  --baseline;
#python api_test.py --dataset devign --features ggnn --lambda1 0 --lambda2 0  --baseline;
#python api_test.py --dataset devign --features wo_ggnn --lambda1 0 --lambda2 0  --baseline --baseline_balance  --baseline_model lr;
python api_test.py --dataset devign --features ggnn --lambda1 0 --lambda2 0  --baseline  --baseline_balance  --baseline_model lr;
#python api_test.py --dataset devign --features wo_ggnn --lambda1 0 --lambda2 0  --baseline --baseline_balance  --baseline_model rf;
#python api_test.py --dataset devign --features ggnn --lambda1 0 --lambda2 0  --baseline  --baseline_balance  --baseline_model rf;