# Run 5 fold cross validation. 
After the 5 fold cross validation, it will out put an model trained on weakly labelled data set under experimentas/kaggle folder
``python

CUDA_VISIBLE_DEVICES=1 python s_train_bilstm_tagger.py --data data/testKaggle2.csv \
--save experiments/kaggle/ \
--epochs 50 --batch-size 300  --partition loo --tags o Y M D h m s --max_len 104 --label R.E.tag --cuda``

# Run pretrain

First, uncomment the ``pretrain_model()`` on line 374, comment the ``datetime_trsize()`` on line 375.

``python

CUDA_VISIBLE_DEVICES=1 python s_train_bilstm_tagger.py --data data/testKaggleAll.csv \
--save experiments/kaggle/pretrain \
--params experiments/kaggle/loo_R.E.tag_best_args.pkl \
--epochs 2 --cuda --batch-size 300 --tags o Y M D h m s --max_len 104 --label R.E.tag ``

# Run fine-tuning 

First, uncomment the  ``
``python

CUDA_VISIBLE_DEVICES=1 python s_train_bilstm_tagger.py --data data/testKaggleAll.csv \
--save experiments/kaggle/pretrain \
--params experiments/kaggle/XXX.pkl \
--epochs 2 --cuda --batch-size 300 --tags o Y M D h m s --max_len 104 --label R.E.tag ``