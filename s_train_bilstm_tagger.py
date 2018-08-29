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
execfile('text_character_bilstm_tagger.py')
# exec(open('./text_character_bilstm_tagger.py').read())
# kaggle_seq_label = pd.read_excel('data/testKaggle2_seq_tolabel_all.xlsx', encoding='utf-8')
# for i in range(len(kaggle_seq_label)):
#     if i % 3 == 0 :
#         kaggle_seq_label.iloc[i, :] = kaggle_seq_label.iloc[i, :].fillna(' ')
#     else:
#         kaggle_seq_label.iloc[i, :] = kaggle_seq_label.iloc[i, :].fillna('o')
        
# seq_label = kaggle_seq_label.iloc[[i for i in range(len(kaggle_seq_label)) if i % 3 == 1], :]
# seq_label = np.array([u''.join(row) for _, row in seq_label.iterrows()])
# strings = kaggle_seq_label.iloc[[i for i in range(len(kaggle_seq_label)) if i % 3 == 0], :]
# strings = np.array([u''.join(row) for _, row in strings.iterrows()])
# str_len = np.array([len(s) for s in strings])
# tag_len = np.array([len(s) for s in seq_label])
# np.all(str_len == tag_len)
# bound_label = kaggle_seq_label.iloc[[i for i in range(len(kaggle_seq_label)) if i % 3 == 2], :]
# bound_label = np.array([u''.join(row) for _, row in bound_label.iterrows()])

# kaggle_data = pd.read_excel('data/testKaggle2_25regex.xlsx', encoding='utf-8')
# kaggle_data.String[kaggle_data.Label == 1] = strings
# kaggle_data['Tag'] = [u'o'*len(r['String']) for _, r in kaggle_data.iterrows()]
# kaggle_data.Tag[kaggle_data.Label == 1] = seq_label
# kaggle_data['Boundary'] = [u'o'*len(r['String']) for _, r in kaggle_data.iterrows()]
# kaggle_data.Boundary[kaggle_data.Label == 1] = bound_label

def package(x_train, y_train, volatile=False):
  """Package data for training / evaluation."""
  dat = map(lambda x: map(lambda y: dictionary.word2idx[y], x.encode('ascii', errors='replace')), x_train)
  y_train = map(lambda x: map(lambda y: tags.word2idx[y], x), y_train)
  # maxlen = 0
  # for item in dat:
  #     maxlen = max(maxlen, len(item))
  maxlen = args.max_len
  # maxlen = min(maxlen, 500)
  for i in range(len(x_train)):
    if maxlen < len(dat[i]):
      dat[i] = dat[i][:maxlen]
      y_train[i] = y_train[i][:maxlen]
    else:
      for j in range(maxlen - len(dat[i])):
        dat[i].append(dictionary.word2idx['<pad>'])
        y_train[i].append(0)
  dat = Variable(torch.LongTensor(dat), volatile=volatile)
  targets = Variable(torch.LongTensor(y_train), volatile=volatile)
  return dat.t(), targets


# def perfs(tags_true, tags_pred):
#   micro_acc = [metrics.f1_score(tags_true[i], tags_pred[i]) for i in range(len(tags_true))]
#   macro_acc = 
#   micro_f1 = 
#   macro_f1 = 

def perfs(tags_true, tags_pred):
  # print(tags_pred[0], tags_true[0], np.mean(tags_true[0] == tags_pred[0]))
  # print(tags_true[0] == tags_pred[0])
  
  pos_baselines = [np.mean(true == 0) for true in tags_true]
  pos_accs = [np.mean(tags_true[i] == tags_pred[i]) for i in range(len(tags_pred))]
  
  whole_accs = [np.all(tags_true[i] == tags_pred[i]) for i in range(len(tags_pred))]
  
          
  # perf = {'Baseline': np.mean(pos_baselines), \
  #         'ACC': np.mean(pos_accs), \
  #         'All': np.mean(whole_accs)}

  perf = {'ACC': np.mean(pos_accs), \
          'All': np.mean(whole_accs)}

  return perf   

def model_selection(data, kf_func):
  kf = kf_func(data)
  model_dir = '%s/%s/' %(args.save, kf_func.__name__)
  for name in ['lr', 'nlayers', 'nhid', 'emsize']:
    model_dir += '%s-%s_' %(name, vars(args)[name])

  
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    

  table = []
  fold = 0
  xs = []
  ys = []
  preds = [] 
  
  for train_idx, test_idx in kf:
    x_train = data.iloc[train_idx, :]['String'].values
    y_train = data.iloc[train_idx, :]['Tag'].values
    x_test = data.iloc[test_idx, :]['String'].values
    y_test = data.iloc[test_idx, :]['Tag'].values
    
    
    print('#train = %s, #test = %s' %(len(x_train), len(x_test)))
    
    model, tags_pred, tags_true, perf = train_test_tagger(x_train, y_train, x_test, y_test, args)

    xs += x_test.tolist()
    ys += tags_true.tolist()
    preds += tags_pred.tolist()
    table.append(perf)
    fold = fold + 1
    with open(os.path.join(model_dir, 'model_fold_%s.pt' %(fold)), 'wb') as f:
      torch.save(model, f)
    f.close()  

  pd.to_pickle(args, os.path.join(model_dir, 'args.pkl'))
  pd.to_pickle({'tag2idx':tags, 'preds':preds, 'labels':ys, 'strings':xs}, os.path.join(model_dir, \
             'meta-%s.pkl' %(int(time.time()))))
  table = pd.DataFrame(table, index = ['Fold %s' %(i) for i in range(5)])
  ave = table.mean(axis=0)
  ave.name = 'Average'
  table = table.append(ave)
  
  overall = pd.Series(perfs(ys, preds))
  overall.name='Overall'
  table = table.append(overall)
  print(table)
  return table  
 

def get_results(kaggle_data):
  flname = args.label
#  kaggle_records = pd.read_pickle('%s/%s_%s_pred.pkl' %(args.save, args.partition, flname))
  kaggle_records = []
  for i in range(10):
    lr = 1.0 / (2** np.random.random_integers(6, 13))
    emsize = np.random.choice([20, 30, 40, 50, 60, 70, 80])
    nhid =  np.random.choice([50, 75, 100, 125, 150, 200])
    nlayers = np.random.choice([1, 2, 5])
    args.lr = lr
    args.emsize = emsize
    args.nhid = nhid
    args.nlayers = nlayers
    
    record = {'args':copy.deepcopy(args)}  

    print("Training with %s" %(str(record)))
    if args.partition == 'loo':
      table1 = model_selection(kaggle_data, loo_part)
      record['loo'] = table1
      record['loo_acc'] = table1.ix['Overall', 'ACC']

    if args.partition == 'random':
      table2 = model_selection(kaggle_data, random_part)    
      record['random'] = table2
      record['random_acc'] = table2.ix['Overall', 'ACC']

    kaggle_records.append(record)
    if not os.path.exists('%s/'%(args.save)):
      os.makedirs('%s'%(args.save))
    
    
    pd.to_pickle(kaggle_records, '%s/%s_%s_pred.pkl' %(args.save, args.partition, flname)) 

  accs = np.array([r['%s_acc' %(args.partition)] for r in kaggle_records])
  best_perf = np.where(accs == np.max(accs))[0][0]
#  print(kaggle_records[best_perf]['loo'])
  print(kaggle_records[best_perf][args.partition])

  print('Start training the best model.....')
  best_args = kaggle_records[best_perf]['args']
  x_train = kaggle_data['String'].values
  y_train = kaggle_data['Tag'].values
  model, _, _, _ = train_test_tagger(x_train, y_train, x_train[:10], y_train[:10], best_args)

  with open(os.path.join(args.save, '%s_%s_best.pt'%(args.partition, flname)), 'wb') as f:
    torch.save(model.state_dict(), f)
  f.close()
  pd.to_pickle(best_args, os.path.join(args.save, '%s_%s_best_args.pkl'%(args.partition, flname)))




#%%
def tagger():
  kaggle_data = pd.read_csv(args.data, encoding='utf-8')
  kaggle_data['Tag'] = kaggle_data[args.label]
  # kaggle_data.columns = ['Outlet', 'kaggle_time', 'title', 'url', 'String', 'Prediction',
  #        u'boundary labels', u'Tag']
  # kaggle_data.Outlet.fillna(method='ffill', inplace=True)      
  # kaggle_data.title.fillna(method='ffill', inplace=True)      
  # kaggle_data.kaggle_time.fillna(method='ffill', inplace=True)
  # kaggle_data.url.fillna(method='ffill', inplace=True)
  
  # kaggle_data = kaggle_data[~ pd.isnull(kaggle_data.Tag)]
  
  str_len = np.array([len(s) for s in kaggle_data.String])
  tag_len = np.array([len(s) for s in kaggle_data.Tag])
  assert(np.all(str_len == tag_len))
  seq_data = kaggle_data[str_len == tag_len] 
  get_results(seq_data)
  

  
def tagger_trsize():
  kaggle_data = pd.read_csv(args.data, encoding='utf-8')
  kaggle_data['Tag'] = kaggle_data[args.label]
  # kaggle_data.columns = ['Outlet', 'kaggle_time', 'title', 'url', 'String', 'Prediction',
  #        u'boundary labels', u'Tag']
  # kaggle_data.Outlet.fillna(method='ffill', inplace=True)      
  # kaggle_data.title.fillna(method='ffill', inplace=True)      
  # kaggle_data.kaggle_time.fillna(method='ffill', inplace=True)
  # kaggle_data.url.fillna(method='ffill', inplace=True)  
  # kaggle_data = kaggle_data[~ pd.isnull(kaggle_data.Tag)]

  str_len = np.array([len(s) for s in kaggle_data.String])
  tag_len = np.array([len(s) for s in kaggle_data.Tag])
  assert(np.all(str_len == tag_len))
  seq_data = kaggle_data[str_len == tag_len]
  if not os.path.exists(args.save):
    os.makedirs(args.save)  

  flname = args.label
  # ts_record = pd.read_pickle('%s/%s_%s_trsize_%s.pkl' %(args.save, \
  #                 args.partition, flname, args.fold))
  # train_idx, test_idx = ts_record['train_idx'], ts_record['test_idx']
  # x_test = seq_data.ix[train_idx, 'String'].values
  # y_test = seq_data.ix[test_idx, 'Tag'].values
  # tr_perfs = ts_record['perfs']
  # tr_preds = ts_record['preds']
  # tr_size = ts_record['tr_size']

  if args.partition == 'random':
    kf = list(random_part(seq_data))
  else:
    kf = loo_part(seq_data)
    
  train_idx, test_idx = kf[args.fold][0], kf[args.fold][1]
  print(len(train_idx), len(test_idx))
  x_test = seq_data.iloc[test_idx, :]['String'].values
  y_test = seq_data.iloc[test_idx, :]['Tag'].values
  tr_perfs = []
  tr_preds = []
  if (len(train_idx) < 10000):
    tr_size = [20, 50, 100, 200, 500, 1000, 2000, len(train_idx)]
  else:
    tr_size = [200, 500, 1000, 2000, 5000, 10000, len(train_idx)]
  # tr_size = [len(train_idx)]
  # ts_record = {'x_test':x_test, 'y_test':y_test, 'tag2idx':tags, 'tr_size':tr_size}
  ts_record = {'tag2idx':tags, 'tr_size':tr_size, 'train_idx':kaggle_data.index.values[train_idx],\
               'test_idx':kaggle_data.index.values[test_idx]}

  _, tags_true = package(x_test, y_test, volatile=True)
  tags_true = tags_true.data.numpy()
  ts_record['tags_true'] = tags_true

  if args.pretrain != '':
    # Baseline
    baseline = np.zeros(tags_true.shape)
    tr_perfs.append(perfs(tags_true, baseline))
    tr_preds.append(baseline)

    # Regular Expression
    re_test = seq_data.ix[test_idx, 'R.E.tag'].values
    _, tags_re = package(x_test, re_test, volatile=True)
    tags_re = tags_re.data.numpy()
    tr_perfs.append(perfs(tags_true, tags_re))
    tr_preds.append(tags_re)

    # Pretrained Regular Expression
    pretrain_pred = pretrain_predict(x_test, y_test, args)
    tr_perfs.append(perfs(tags_true, pretrain_pred))
    tr_preds.append(pretrain_pred)

  for i, ts in enumerate(tr_size):
    np.random.seed(i)
    train_idx_s = np.random.choice(train_idx, ts, replace=False)
    x_train = seq_data.iloc[train_idx_s, ]['String'].values
    y_train = seq_data.iloc[train_idx_s, ]['Tag'].values
    print('Train size = %s, test size = %s' %(len(x_train), len(x_test)))    
    model, tags_pred, _, perf = train_test_tagger(x_train, y_train, x_test, y_test, args)
    
    print(perf)
    tr_perfs.append(perf)
    tr_preds.append(tags_pred)
    
    ts_record['preds'] = tr_preds
    ts_record['perfs'] = tr_perfs
    pd.to_pickle(ts_record, '%s/%s_%s_trsize_%s.pkl' %(args.save, args.partition, flname, args.fold))


def pretrain_model():
  kaggle_data = pd.read_csv(args.data, encoding='utf-8')
  kaggle_data['Tag'] = kaggle_data[args.label]
#  kaggle_data.columns = ['Outlet', 'kaggle_time', 'title', 'url', 'String', 'Prediction',
#         u'boundary labels', u'Tag']
#  kaggle_data.Outlet.fillna(method='ffill', inplace=True)      
#  kaggle_data.title.fillna(method='ffill', inplace=True)      
#  kaggle_data.kaggle_time.fillna(method='ffill', inplace=True)
#  kaggle_data.url.fillna(method='ffill', inplace=True)
  flname = args.label
  kaggle_data = kaggle_data[~ pd.isnull(kaggle_data.Tag)]

  str_len = np.array([len(s) for s in kaggle_data.String])
  tag_len = np.array([len(s) for s in kaggle_data.Tag])
  assert(np.all(str_len == tag_len))
  seq_data = kaggle_data[str_len == tag_len]

  if not os.path.exists(args.save):
    os.makedirs(args.save)  
  x_train = seq_data['String'].values
  y_train = seq_data['Tag'].values
  print(x_train.shape)
  model, _, _, _ = train_test_tagger(x_train, y_train, x_train[:1000], y_train[:1000], args)
  with open(os.path.join(args.save, '%s_%s_pretrain_%s.pt'%(flname, args.data.split('/')[-1][:-4], args.epochs)), 'wb') as f:
    torch.save(model.state_dict(), f)
  f.close()
  pd.to_pickle(args, os.path.join(args.save, '%s_%s_pretrain_args.pkl'%(flname, args.data.split('/')[-1][:-4]))) 

if __name__ == '__main__':
  
  args = get_args()
  dictionary = Dictionary()
  dictionary.idx2word = ['<pad>'] + [chr(i) for i in range(256)]
  dictionary.word2idx = {w:i for i, w in enumerate(dictionary.idx2word)}
  n_token = len(dictionary)

  tags = Dictionary()
  tags.idx2word = [tag for tag in args.tags]
  tags.word2idx = {tag:i for i, tag in enumerate(tags.idx2word)}
  print(tags.idx2word, tags.word2idx)

  torch.manual_seed(args.seed)
  if torch.cuda.is_available():
    if not args.cuda:
      print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
      torch.cuda.manual_seed(args.seed)
  random.seed(args.seed)    

  if args.params != '':
    kaggle_records = pd.read_pickle(args.params)
    if type(kaggle_records) == list:
      accs = np.array([r['%s_acc' %(args.partition)] for r in kaggle_records])
      best_perf = np.where(accs == np.max(accs))[0][0]
      args.emsize = kaggle_records[best_perf]['args'].emsize
      args.lr = kaggle_records[best_perf]['args'].lr
      args.nhid = kaggle_records[best_perf]['args'].nhid  
      args.nlayers = kaggle_records[best_perf]['args'].nlayers
    else:
      args.emsize = kaggle_records.emsize
      args.nhid = kaggle_records.nhid
      args.nlayers = kaggle_records.nlayers
      # args.dropout = kaggle_records.dropout
      # args.max_len = kaggle_records.max_len
      args.lr = kaggle_records.lr
    
    # pretrain_model()
    tagger_trsize()
  else:
    tagger()
