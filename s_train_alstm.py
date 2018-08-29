# -*- coding: utf-8 -*-

#CUDA_VISIBLE_DEVICES=0 python s_train_alstm.py --pooling conv --epochs 50 --batch-size 300 --cuda

from text_classification_selfattention_bilstm import train_test_sbl
from datautil.util import Dictionary, get_args, random_part, loo_part
import pandas as pd
import numpy as np
import os
import torch
import random
from sklearn import metrics
from sklearn.cross_validation import KFold
import time
import matplotlib.pylab as plt
import seaborn as sns
from torch.autograd import Variable
import copy

execfile('text_classification_selfattention_bilstm.py')
def Frobenius(mat):
  size = mat.size()
  if len(size) == 3:  # batched matrix
    ret = (torch.sum(torch.sum((mat ** 2), 1, keepdim=True), 2, keepdim=True).squeeze() + 1e-10) ** 0.5
    return torch.sum(ret) / size[0]
  else:
    raise Exception('matrix for computing Frobenius norm should be with 3 dims')

def package(x_train, y_train, volatile=False):
  """Package data for training / evaluation."""
  dat = map(lambda x: map(lambda y: dictionary.word2idx[y], x.encode('ascii', errors='replace')), x_train)
  # maxlen = 0
  # for item in dat:
  #     maxlen = max(maxlen, len(item))
  maxlen = args.max_len
  # maxlen = min(maxlen, 500)
  for i in range(len(x_train)):
    if maxlen < len(dat[i]):
      dat[i] = dat[i][:maxlen]
    else:
      for j in range(maxlen - len(dat[i])):
        dat[i].append(dictionary.word2idx['<pad>'])
  dat = Variable(torch.LongTensor(dat), volatile=volatile)
  targets = Variable(torch.LongTensor(y_train), volatile=volatile)
  return dat.t(), targets

def model_selection(data, kf_func, args):
  kf = kf_func(data)
  
  
  model_dir = '%s/%s/' %(args.save, kf_func.__name__)
  if args.pooling == 'all':
    model_dir += 'alstm/'        
    for name in ['attention_hops', 'lr', 'penalization_coeff', 'nfc', 'nhid']:
      model_dir += '%s-%s_' %(name, vars(args)[name])
  elif args.pooling == 'max' or args.pooling == 'mean':
    model_dir += 'lstm/'
    for name in ['lr', 'nfc', 'nhid']:
      model_dir += '%s-%s_' %(name, vars(args)[name])

  elif args.pooling == 'conv':
      model_dir += 'cnn/'
      for name in ['lr', 'nfc']:
        model_dir += '%s-%s_' %(name, vars(args)[name])
      model_dir += '%s-%s' %('ksize', args.filter_nums[0])
      
  elif args.pooling == 'mlp':
    model_dir += 'mlp/'
    model_dir += 'lr-%s_nfc-%s' %(vars(args)['lr'], vars(args)['nfc'])
  
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  table = []
  fold = 0
  chars = np.zeros((len(data), args.max_len), dtype=np.int32)
  classes = np.zeros((len(data),))
  atts = np.zeros((len(data), args.attention_hops, args.max_len))
  for train_idx, test_idx in kf:
    x_train = data.iloc[train_idx, :]['String'].values
    y_train = data.iloc[train_idx, :]['Label'].values
    x_test = data.iloc[test_idx, :]['String'].values
    y_test = data.iloc[test_idx, :]['Label'].values
    
    
    print('#train = %s, #test = %s, pos frac = %s' %(len(y_train), len(y_test), np.mean(y_test)))
    
    model, perf, datas, attentions, y_predicted = train_test_sbl(x_train, y_train, x_test, y_test, args)
    chars[test_idx, :] = datas
    atts[test_idx, :] = attentions
    classes[test_idx] = y_predicted
    table.append(perf)
    fold = fold + 1
    with open(os.path.join(model_dir, 'model_fold_%s.pt' %(fold)), 'wb') as f:
      torch.save(model.state_dict(), f)
    f.close()
    
  #   if args.pooling == 'all':
  #     plot_path = os.path.join(model_dir, 'plot_fold_%02d.png' %(fold))
  #     plot_att(datas, y_predicted, y_test, attentions, plot_path)

  # if args.pooling == 'all':
  #   plot_path = os.path.join(model_dir, 'plot_all.png')
  #   plot_att(chars, classes, data.Label.values, atts, plot_path)

    
  pd.to_pickle(args, os.path.join(model_dir, 'args.pkl'))
  pd.to_pickle({'preds':classes, 'labels':data.Label.values, \
             'atts':atts, 'chars':chars}, os.path.join(model_dir, \
             'meta-%s.pkl' %(int(time.time()))))
  table = pd.DataFrame(table, index = ['Fold %s' %(i) for i in range(5)])
  ave = table.mean(axis=0)
  ave.name = 'Average'
  table = table.append(ave)
  
  overall = pd.Series(perfs(data.Label.values, classes))
  overall.name='Overall'
  table = table.append(overall)
  print(table)
  return table

#%% 
def get_results(kaggle_data):
  kaggle_records = []
  pooling = {'max':'lstm', 'all':'alstm', 'conv':'cnn', 'mlp':'mlp'}
  flname = pooling[args.pooling]
  for i in range(10):
    lr = 1.0 / (2** np.random.random_integers(6, 13))
    # penal = 1.0 / (2** np.random.random_integers(0, 6))
    # lr = np.random.choice([0.001, 0.002, 0.005])
    hops = np.random.choice([20, 25, 30])
    penal = 1.0 / (2** np.random.random_integers(0, 5))
    ksize = np.random.choice([20, 30, 40, 50, 60, 70, 80])
    nhid = np.random.choice([50, 75, 100, 125, 150, 200])
    # nhid = np.random.choice([100, 200, 300])
    nfc = np.random.choice([20, 50, 100, 200, 500])
    # nfc = [20, 50, 100, 200, 300, 500][i]
    # nfc = np.random.choice([300, 512])
    # nhid = 300
    args.lr = lr
    args.attention_hops = hops
    args.penalization_coeff = penal
    args.filter_nums = [ksize] * len(args.filters)
    args.nhid = nhid
    args.nfc = nfc
    

    # args.lr = 0.001953125
    # args.attention_hops = 30
    # args.penalization_coeff = 0.125
    # args.filter_nums = [60] * len(args.filters)
    # args.nhid = 300
    # args.nfc = 200
    
    # record = {'lr':lr, 'hops':hops, 'penal':penal,\
    #           'nhid':nhid, 'nfc':nfc, 'ksize':ksize} 
    record = {'args':copy.deepcopy(args)}

    print("Training with %s" %(str(record)))
    if args.partition == 'loo':
      table1 = model_selection(kaggle_data, loo_part, args)
      record['loo'] = table1
      record['loo_acc'] = table1.ix['Overall', 'ACC']

    if args.partition == 'random':
      table2 = model_selection(kaggle_data, random_part, args)    
      record['random'] = table2
      record['random_acc'] = table2.ix['Overall', 'ACC']

    kaggle_records.append(record)
    if not os.path.exists('%s/'%(args.save)):
      os.makedirs('%s'%(args.save))
    
    
    pd.to_pickle(kaggle_records, '%s/%s_%s_pred.pkl' %(args.save, args.partition, flname)) 
    
  accs = np.array([r['%s_acc' %(args.partition)] for r in kaggle_records])
  best_perf = np.where(accs == np.max(accs))[0][0]
  print(kaggle_records[best_perf][args.partition])

  print('Start training the best model.....')
  best_args = kaggle_records[best_perf]['args']
  x_train = kaggle_data['String'].values
  y_train = kaggle_data['Label'].values
  model, _, _, _, _  = train_test_sbl(x_train, y_train, x_train[:10], y_train[:10], best_args)
  with open(os.path.join(args.save, '%s_%s_best.pt'%(args.partition, flname)), 'wb') as f:
    torch.save(model.state_dict(), f)
  f.close()
  pd.to_pickle(best_args, os.path.join(args.save, '%s_%s_best_args.pkl'%(args.partition, flname)))



def plot_att(data, pred, label, attention, save):
  chars = np.array([[dictionary.idx2word[idx] if idx != 0 else ' ' for idx in sent] for sent in data])
  fig, ax = plt.subplots(figsize=(30, 10))
  a = attention.sum(axis=1)
  a = a / a.sum(axis=1, keepdims=True)
  to_plot = []
#  print(pd.Series(label).value_counts())
  for i in range(len(np.unique(label))):
    pos = np.where((pred==label) & (label==i))[0]
    to_plot.extend(list(pos[:10]))
  sns.heatmap(data=a[to_plot,:], annot=chars[to_plot,:], fmt='s', linewidths=0, cmap='Reds', \
              vmin=0, vmax=min([a[to_plot,:].max(), 0.1]))
  plt.tight_layout()
  plt.savefig(save, bbox_inches='tight')

  
def plot_att2(data, pred, label, attention, save):
  chars = np.array([[dictionary.idx2word[idx] if idx != 0 else ' ' for idx in sent] for sent in data])
  a = attention.sum(axis=1)
  a = a / a.sum(axis=1, keepdims=True)
  sns.heatmap(data=a, annot=chars, fmt='s', linewidths=0, cmap='Reds', \
              vmin=0, vmax=min([a.max(), 0.1]), cbar=False)
  plt.tight_layout()
  plt.savefig(save, bbox_inches='tight')

def perfs(y_true, y_classes):
  if len(y_true) == 0: return {}
  labels = range(1, y_true.max()+1)
  # labels = range(y_true.max())
  # labels = [1]
  perf = {}
  perf['ACC'] = metrics.accuracy_score(y_true, y_classes)   # accuracy
#  perf['AUC']  = metrics.roc_auc_score(y_true, y_probs)        # auc
  perf['Prec']  = metrics.precision_score(y_true, y_classes, labels=labels, average='micro') # precision
  perf['Rec']  = metrics.recall_score(y_true, y_classes, labels=labels, average='micro') # recall
  perf['F1'] = metrics.f1_score(y_true, y_classes, labels=labels, average='micro')
  perf['Baseline']  = 1 - np.mean(y_true != 0)             # naive acc

  print(perf)  
  return perf


#%%
def pred_datetime():
  if args.data[-3:] == 'csv':
    kaggle_data = pd.read_csv(args.data, encoding='utf-8')
  else:
    kaggle_data = pd.read_excel(args.data, encoding='utf-8')
  kaggle_data.Outlet.fillna(method='ffill', inplace=True)
  # kaggle_data.Outlet = kaggle_data.Outlet.astype(int)
  kaggle_data.String = kaggle_data.String
  # kaggle_data = kaggle_data.sample(len(kaggle_data), random_state=0, replace=False)
  kaggle_data['Label'] = kaggle_data[args.label]
  args.class_number = 2

  get_results(kaggle_data)

def pretrain_model():
  if args.data[-3:] == 'csv':
    kaggle_data = pd.read_csv(args.data, encoding='utf-8')
  else:
    kaggle_data = pd.read_excel(args.data, encoding='utf-8')
  kaggle_data.Outlet.fillna(method='ffill', inplace=True)
  # kaggle_data.String = kaggle_data.String.astype(str)
  kaggle_data['Label'] = kaggle_data[args.label]

  args.class_number = 2
  pooling = {'max':'lstm', 'all':'alstm', 'conv':'cnn', 'mlp':'mlp'}
  flname = pooling[args.pooling]
  if not os.path.exists(args.save):
    os.makedirs(args.save)  
  x_train = kaggle_data['String'].values
  y_train = kaggle_data['Label'].values.astype(int)
  print(x_train.shape)
  model, _, _, _, _  = train_test_sbl(x_train, y_train, x_train[:1000], y_train[:1000], args)
  with open(os.path.join(args.save, '%s_%s_pretrain.pt'%(flname, args.data.split('/')[-1][:-5])), 'wb') as f:
    torch.save(model.state_dict(), f)
  f.close()
  pd.to_pickle(args, os.path.join(args.save, '%s_%s_pretrain_args.pkl'%(flname, args.data.split('/')[-1][:-5]))) 

def datetime_trsize():
  if args.data[-3:] == 'csv':
    kaggle_data = pd.read_csv(args.data, encoding='utf-8')
  else:
    kaggle_data = pd.read_excel(args.data, encoding='utf-8')
  kaggle_data.Outlet.fillna(method='ffill', inplace=True)
  kaggle_data['Label'] = kaggle_data[args.label]
  pooling = {'max':'lstm', 'all':'alstm', 'conv':'cnn', 'mlp':'mlp'}
  flname = pooling[args.pooling]  

  if not os.path.exists(args.save):
    os.makedirs(args.save)  
  # seq_data['Tag'] = new_tags  
  if args.partition == 'random':
    kf = list(random_part(kaggle_data))
  else:
    kf = loo_part(kaggle_data)
    
    
  train_idx, test_idx = kf[args.fold][0], kf[args.fold][1]
  args.class_number = 2
  x_test = kaggle_data.iloc[test_idx, :]['String'].values
  y_test = kaggle_data.iloc[test_idx, :]['Label'].values.astype(int)
  tr_perfs = []
  tr_preds = []
  if args.pretrain != '':
    pretrain_pred = pretrain_predict(x_test, y_test, args)
    tr_perfs.append(perfs(y_test, pretrain_pred))
    tr_preds.append(pretrain_pred)

  if (len(train_idx) < 100000):
    tr_size = [20, 50, 100, 200, 500, 1000, 2000, len(train_idx)]
  else:
    tr_size = [200, 500, 1000, 2000, 5000, 10000, len(train_idx)]
  # tr_size = [len(train_idx)]
  ts_record = {'tr_size':tr_size, 'train_idx':kaggle_data.index.values[train_idx],\
               'test_idx':kaggle_data.index.values[test_idx]}

  y_train = kaggle_data.iloc[train_idx, :]['Label'].values.astype(int)
  train_pos = train_idx[y_train == 1]
  train_neg = train_idx[y_train != 1]
  # ts_record = {'tr_size':tr_size, 'test_idx': test_idx, 'train_idx':}
  for i, ts in enumerate(tr_size):
    np.random.seed(i)
    # pos_idx = np.random.choice(train_pos, min([int(ts * .5), len(train_pos)]), replace=False)
    # neg_idx = np.random.choice(train_neg, ts - len(pos_idx), replace=False)
    # train_idx_s = np.array(list(pos_idx) + list(neg_idx))
    train_idx_s = np.random.choice(train_idx, ts, replace=False)
    x_train = kaggle_data.iloc[train_idx_s, ]['String'].values
    y_train = kaggle_data.iloc[train_idx_s, ]['Label'].values.astype(int)

    print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
    model, perf, _, _, y_classes = train_test_sbl(x_train, y_train, x_test, y_test, args)
    
    print(perf)
    tr_perfs.append(perf)
    tr_preds.append(y_classes)
    
    ts_record['preds'] = tr_preds
    ts_record['perfs'] = tr_perfs
    pd.to_pickle(ts_record, '%s/%s_%s_trsize_%s.pkl' %(args.save, args.partition, flname, args.fold))


if __name__ == '__main__':
  args = get_args()
  dictionary = Dictionary()
  dictionary.idx2word = ['<pad>'] + [chr(i) for i in range(256)]
  dictionary.word2idx = {w:i for i, w in enumerate(dictionary.idx2word)}
  n_token = len(dictionary)

  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    if not args.cuda:
      print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
      torch.cuda.manual_seed(args.seed)
  random.seed(args.seed)    
  
  if args.params != '':
    # save = args.save
    # data = args.data
    # epochs = args.epochs
    # pretrain = args.pretrain
    kaggle_records = pd.read_pickle(args.params)
    if type(kaggle_records) == list:
      accs = np.array([r['%s_acc' %(args.partition)] for r in kaggle_records])
      best_perf = np.where(accs == np.max(accs))[0][0]
      best_args = kaggle_records[best_perf]['args']
      args.lr = best_args.lr
      args.attention_hops = best_args.attention_hops
      args.penalization_coeff = best_args.penalization_coeff
      args.filter_nums = best_args.filter_nums
      args.nhid = best_args.nhid
      args.nfc = best_args.nfc
    else:
      args.lr = kaggle_records.lr
      args.attention_hops = kaggle_records.attention_hops
      args.penalization_coeff = kaggle_records.penalization_coeff
      args.filter_nums = kaggle_records.filter_nums
      args.nhid = kaggle_records.nhid
      args.nfc = kaggle_records.nfc

    # pretrain_model()
    datetime_trsize()
  else:
    pred_datetime()

       