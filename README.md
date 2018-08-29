
# Get Started

REGuider contains code and datasets for EMNLP 2018 paper:

* **Zhang, S., He, L., Vucetic, S., Dragut, E., Regular Expression Guided Entity Mention Mining from Noisy Web Data, EMNLP, 2018**

To run the code, the following environment is required:
* python==2.7.6
* torch==0.3.1

# Run 5 fold cross validation. 
The 5-fold cross validation is used to select the best hyperparameters based on the weaklly labelled data, via ``random search`` technique. 
After the 5 fold cross validation, the best hyperparameter ``XX.pkl`` is output to the ``experimentas/kaggle`` folder.

``
CUDA_VISIBLE_DEVICES=1 python s_train_alstm.py --data data/XX.csv \
-save experiment/kaggle/ --pooling max --epochs 20 --batch-size 300  --partition random \
--cuda --max_len 104 --label R.E.26``

# Pretrain on weakly labelled data
First, comment the ``datetime_trsize()`` on line 375, uncomment the ``pretrain_model()`` on line 374.
Then run:

``CUDA_VISIBLE_DEVICES=0 python s_train_alstm.py --data data/testKaggleAll.csv \
--save experiment/kaggle/pretrain/ \
--params experiment/kaggle/XX.pkl \
--epochs 2 --cuda --pooling max --batch-size 300 --max_len 104 --label R.E.26
``

# Evaluate the model trained on human labels with pretrain
First, uncomment the ``datetime_trsize()`` on line 375, comment the ``pretrain_model()`` on line 374.
With ``--pretrain`` set to the pretrained model, the ``datetime_trsize()`` will evaluate the model trained on weakly labeled data using diffrent training size. 


``CUDA_VISIBLE_DEVICES=0 python s_train_alstm.py --data data/testKaggleAll.csv \
  --save experiment/kaggle/pretrain \
  --params experiment/kaggle/XX.pkl \
  --epochs 50 --cuda --pooling max --partition loo --batch-size 300 --max_len 104 --label Label --fold 1 \
  --pretrain experiment/kaggle/XX.pt
``

# Evaluate the model trained on human labels without pretrain

``CUDA_VISIBLE_DEVICES=0 python s_train_alstm.py --data data/testKaggleAll.csv \
  --save experiment/kaggle/pretrain \
  --params experiment/kaggle/XX.pkl \
  --epochs 50 --cuda --pooling max --partition loo --batch-size 300 --max_len 104 --label Label --fold 1
``