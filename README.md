
# How to Run 5 fold cross validation: after the 5 fold cross validation, it will out put an model trained on all data set under experimentas/kaggle folder
CUDA_VISIBLE_DEVICES=1 python s_train_bilstm_tagger.py --data data/testKaggle2.csv \
--save experiments/kaggle/ \
--epochs 50 --batch-size 300  --partition loo --tags o Y M D h m s --max_len 104 --label R.E.tag --cuda

# How to run pretrain
# Step 1: uncomment the pretrain_model() on line 371

CUDA_VISIBLE_DEVICES=1 python s_train_bilstm_tagger.py --data data/testKaggleAll.csv \
--save experiments/kaggle/pretrain \
--params experiments/kaggle/loo_R.E.tag_best_args.pkl \
--epochs 2 --cuda --batch-size 300 --tags o Y M D h m s --max_len 104 --label R.E.tag 