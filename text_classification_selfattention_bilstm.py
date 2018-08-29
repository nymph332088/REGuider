# CUDA_VISIBLE_DEVICES=0 python text_classification_selfattention_bilstm.py --pooling conv --epochs 50 --batch-size 300 --cuda --pred datetime
from __future__ import print_function
from model.SBLClassifier import *

from datautil.util import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import json
import time
import random
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import KFold
import time
import matplotlib.pylab as plt

#%%
def train_test_sbl(x_train, y_train, x_test, y_test, args):
  model = Classifier({
      'dropout': args.dropout,
      'ntoken': n_token,
      'nlayers': args.nlayers,
      'nhid': args.nhid,
      'ninp': args.emsize,
      'pooling': args.pooling,
      'attention-unit': args.attention_unit,
      'attention-hops': args.attention_hops,
      'filters': args.filters,
      'filter_nums': args.filter_nums,
      'nfc': args.nfc,
      'dictionary': dictionary,
      'word-vector': args.word_vector,
      'class-number': args.class_number,
      'max_len': args.max_len
      
  })
  if args.cuda:
    model = model.cuda()

  
  if args.pretrain != '':
    print("Loading pre-trained models.....")
    pretrained_dict = torch.load(args.pretrain)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

  print(args)
  I = Variable(torch.zeros(args.batch_size, args.attention_hops, args.attention_hops))
  for i in range(args.batch_size):
    for j in range(args.attention_hops):
      I.data[i][j][j] = 1
  if args.cuda:
    I = I.cuda()

  criterion = nn.CrossEntropyLoss()
  if args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
  elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
  else:
    raise Exception('For other optimizers, please add it yourself. '
                      'supported ones are: SGD and Adam.')


  for epoch in range(args.epochs):
    start_time = time.time()    
    total_loss, total_pure_loss = train(model, x_train, y_train, I, criterion, optimizer, args)
    elapsed = time.time() - start_time
    loss, data, attention, y_classes = evaluate(model, x_test, y_test, criterion, args)
    perf = perfs(y_test, y_classes)
    print('| epoch {:3d} | ms/epoch {:5.2f} | loss {:5.4f} | pure loss {:5.4f}'.format(
        epoch, elapsed, total_loss, total_pure_loss))



  return model, perf, data, attention, y_classes

def train(model, x_train, y_train, I, criterion, optimizer, args):
  model.train()
  total_loss = 0
  total_pure_loss = 0  # without the penalization term
  for batch, i in enumerate(range(0, len(x_train), args.batch_size)):

    start_inds, end_inds = i, min(len(x_train), i+args.batch_size)
    ## data size: [len, bsz]
    data, targets = package(x_train[start_inds:end_inds], y_train[start_inds:end_inds], volatile=False)
    if args.cuda:
        data = data.cuda()
        targets = targets.cuda()
    hidden = model.init_hidden(data.size(1))
    output, attention = model.forward(data, hidden)
    loss = criterion(output.view(data.size(1), -1), targets)
    total_pure_loss += loss.data
    
    if model.mode == 'alstm':  # add penalization term
      attentionT = torch.transpose(attention, 1, 2).contiguous()
      extra_loss = Frobenius(torch.bmm(attention, attentionT) - I[:attention.size(0)])
      loss += args.penalization_coeff * extra_loss
    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm(model.parameters(), args.clip)
    optimizer.step()

    total_loss += loss.data
    if batch % 10 == 0: print('Finished %s out of %s batches' %(batch, len(x_train) // args.batch_size))
  # return total_loss[0] / (len(x_train) // args.batch_size), total_pure_loss[0] / (len(x_train) // args.batch_size)
  return total_loss[0], total_pure_loss[0]

def evaluate(model, x_test, y_test, criterion, args):
  """evaluate the model while training"""
  model.eval()  # turn on the eval() switch to disable dropout
  total_loss = 0
  total_correct = 0
  datas = []
  if model.mode != 'alstm':
    attentions = None
  else:
    attentions = [] 
  preds = []
  for batch, i in enumerate(range(0, len(x_test), args.batch_size)):
    start_inds, end_inds = i, min(len(x_test), i+args.batch_size)
    data, targets = package(x_test[start_inds:end_inds], y_test[start_inds:end_inds], volatile=True)
    if args.cuda:
        data = data.cuda()
        targets = targets.cuda()
    hidden = model.init_hidden(data.size(1))
    output, attention = model.forward(data, hidden)
    output_flat = output.view(data.size(1), -1)
    total_loss += criterion(output_flat, targets).data
    prediction = torch.max(output_flat, 1)[1]
    total_correct += torch.sum((prediction == targets).float())
    datas.append(data.cpu().data.numpy().T)
    if model.mode == 'alstm': attentions.append(attention.cpu().data.numpy())
    preds.append(prediction.cpu().data.numpy())  
    print('Finished %s out of %s batches' %(batch, len(x_test) // args.batch_size))

  datas = np.vstack(datas)
  if model.mode == 'alstm': attentions = np.vstack(attentions)
#  print(preds[0].shape)
  preds = np.hstack(preds)
  # return total_loss[0] / (len(x_test) // args.batch_size), datas, attentions, preds
  return total_loss, datas, attentions, preds

def pretrain_predict(x_test, y_test, args):
  model = Classifier({
      'dropout': args.dropout,
      'ntoken': n_token,
      'nlayers': args.nlayers,
      'nhid': args.nhid,
      'ninp': args.emsize,
      'pooling': args.pooling,
      'attention-unit': args.attention_unit,
      'attention-hops': args.attention_hops,
      'filters': args.filters,
      'filter_nums': args.filter_nums,
      'nfc': args.nfc,
      'dictionary': dictionary,
      'word-vector': args.word_vector,
      'class-number': args.class_number,
      'max_len': args.max_len
      
  })
  if args.cuda:
    model = model.cuda()

  if args.pretrain != '':
    print("Loading pre-trained models.....")
    pretrained_dict = torch.load(args.pretrain)
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
  model.eval()
  data, targets = package(x_test, y_test, volatile=True)
  if args.cuda:
      data = data.cuda()
      targets = targets.cuda()
  hidden = model.init_hidden(data.size(1))
  output, attention = model.forward(data, hidden)
  output_flat = output.view(data.size(1), -1)
  prediction = torch.max(output_flat, 1)[1]
  prediction = prediction.cpu().data.numpy()  
  return prediction

def pred_day(args):
  kaggle_data = pd.read_excel('data/testKaggle2.xlsx', sheetname='Sheet1')
  kaggle_data = kaggle_data.sample(len(kaggle_data), random_state=0, replace=False)

  kaggle_data.fillna(0, inplace=True)
  LABELS = [0, 5, 24, 25, 26, 27, 28, 29, 30]
  kaggle_data = kaggle_data[kaggle_data.day.isin(LABELS)]
  kaggle_data.Label = np.zeros(len(kaggle_data))
  for i, l in enumerate(LABELS):
    kaggle_data.Label[kaggle_data.day==l] = i
    
  kaggle_data.Label = kaggle_data.Label.astype(int)
  args.class_number = len(LABELS)
  args.save = 'classify_1130/kaggle_day'
  get_results(kaggle_data)
  get_best_var(kaggle_data)



def pred_month(args):
  kaggle_data = pd.read_excel('data/testKaggle2.xlsx', sheetname='Sheet1')
  kaggle_data = kaggle_data.sample(len(kaggle_data), random_state=0, replace=False)
  kaggle_data.fillna(0, inplace=True)
  LABELS = [0, 8, 9, 10]
  kaggle_data = kaggle_data[kaggle_data.month.isin(LABELS)]
  kaggle_data.Label = np.zeros(len(kaggle_data))
  for i, l in enumerate(LABELS):
    kaggle_data.Label[kaggle_data.month==l] = i
  
  kaggle_data.Label = kaggle_data.Label.astype(int)
  args.class_number = len(LABELS)
  args.save = 'classify_1130/kaggle_month'
  get_results(kaggle_data)
  get_best_var(kaggle_data)

def pred_datetime(args):
  kaggle_data = pd.read_excel('data/testKaggle2.xlsx', sheetname='Sheet1')
  kaggle_data = kaggle_data.sample(len(kaggle_data), random_state=0, replace=False)
  args.class_number = 2
  args.save = 'classify_1130/kaggle_datetime'
  get_results(kaggle_data)
  get_best_var(kaggle_data)
  

def pred_year_reg(args):
  reg_data = pd.read_excel('data/Regex Examples with Labels.xlsx', sheetname='Sheet1')
  reg_data = reg_data.sample(len(reg_data), random_state=0, replace=False)
  LABELS = sorted(reg_data.year.unique())
  reg_data['Label'] = np.zeros(len(reg_data))
  for i, l in enumerate(LABELS): reg_data.Label[reg_data.year == l] = i
  reg_data.Label = reg_data.Label.astype(int)
  args.class_number = len(LABELS)
  args.save = 'classify_1130/regex_year'
  get_results(reg_data)
  get_best_var(reg_data)


def pred_month_reg():
  reg_data = pd.read_excel('data/Regex Examples with Labels.xlsx', sheetname='Sheet1')
  reg_data = reg_data.sample(len(reg_data), random_state=0, replace=False)
  LABELS = sorted(reg_data.month.unique())
  reg_data['Label'] = np.zeros(len(reg_data))
  for i, l in enumerate(LABELS): reg_data.Label[reg_data.month == l] = i
  reg_data.Label = reg_data.Label.astype(int)
  args.class_number = len(LABELS)
  args.save = 'classify_1130/regex_month'
  get_results(reg_data)
  get_best_var(reg_data)

def pred_day_reg(args):
  reg_data = pd.read_excel('data/Regex Examples with Labels.xlsx', sheetname='Sheet1')
  reg_data = reg_data.sample(len(reg_data), random_state=0, replace=False)
  LABELS = sorted(reg_data.day.unique())
  reg_data['Label'] = np.zeros(len(reg_data))
  for i, l in enumerate(LABELS): reg_data.Label[reg_data.day == l] = i
  reg_data.Label = reg_data.Label.astype(int)
  args.class_number = len(LABELS)
  args.save = 'classify_1130/regex_day'
  get_results(reg_data)
  get_best_var(reg_data)

def pred_hour_reg(args):
  reg_data = pd.read_excel('data/Regex Examples with Labels.xlsx', sheetname='Sheet1')
  reg_data = reg_data.sample(len(reg_data), random_state=0, replace=False)
  LABELS = sorted(reg_data.hour.unique())
  reg_data['Label'] = np.zeros(len(reg_data))
  for i, l in enumerate(LABELS): reg_data.Label[reg_data.hour == l] = i
  reg_data.Label = reg_data.Label.astype(int)
  args.class_number = len(LABELS)
  args.save = 'classify_1130/regex_hour'
  get_results(reg_data)
  get_best_var(reg_data)

def pred_minute_reg(args):
  reg_data = pd.read_excel('data/Regex Examples with Labels.xlsx', sheetname='Sheet1')
  reg_data = reg_data.sample(len(reg_data), random_state=0, replace=False)
  LABELS = sorted(reg_data.minute.unique())
  reg_data['Label'] = np.zeros(len(reg_data))
  for i, l in enumerate(LABELS): reg_data.Label[reg_data.minute == l] = i
  reg_data.Label = reg_data.Label.astype(int)
  args.class_number = len(LABELS)
  args.save = 'classify_1130/regex_minute'
  get_results(reg_data)
  get_best_var(reg_data)

def get_best_var(kaggle_data, args):
  pooling = {'max':'lstm', 'all':'alstm', 'conv':'cnn', 'mlp':'mlp'}
  flname = pooling[args.pooling]
  results = pd.read_pickle('%s/%s_pred.pkl' %(args.save, flname))
  args.save = '%s_best'%(args.save)
  best_vars = {}
  
  accs = np.array([r['random'].ix['Overall', 'F1'] for r in results])
  best = np.where(accs == accs.max())[0][0]
  lr, hops, penal, nhid, nfc, ksize \
        = tuple([results[best][key] for key in ['lr', 'hops', 'penal', \
          'nhid', 'nfc', 'ksize']])
  
  args.lr = lr
  args.attention_hops = hops
  args.penalization_coeff = penal
  args.filter_nums = [ksize] * len(args.filters)
  args.nhid = nhid
  args.nfc = nfc          
  
  random_reps = {'lr':lr, 'hops':hops, 'penal':penal,\
              'nhid':nhid, 'nfc':nfc, 'ksize':ksize}
  
  print ('Replicates for 10 times')
  random_reps['reps'] = [model_selection(kaggle_data, random_part) \
                        for _ in range(10)]
  
  best_vars['random'] = random_reps
  pd.to_pickle(best_vars, '%s/%s_reps.pkl' %(args.save, flname))
  
  accs = np.array([r['loo'].ix['Overall', 'F1'] for r in results])
  best = np.where(accs == accs.max())[0][0]
  lr, hops, penal, nhid, nfc, ksize \
        = tuple([results[best][key] for key in ['lr', 'hops', 'penal', \
          'nhid', 'nfc', 'ksize']])
  
  args.lr = lr
  args.attention_hops = hops
  args.penalization_coeff = penal
  args.filter_nums = [ksize] * len(args.filters)
  args.nhid = nhid
  args.nfc = nfc          
  
  loo_reps = {'lr':lr, 'hops':hops, 'penal':penal,\
              'nhid':nhid, 'nfc':nfc, 'ksize':ksize}
  loo_reps['reps'] = [model_selection(kaggle_data, loo_part) \
                        for _ in range(10)]

  best_vars['loo'] = loo_reps
  pd.to_pickle(best_vars, '%s/%s_reps.pkl' %(args.save, flname))
                        


# if __name__ == '__main__':
#     # parse the arguments
#   args = get_args()
#   # Set the random seed manually for reproducibility.  
#   torch.manual_seed(args.seed)
#   if torch.cuda.is_available():
#       if not args.cuda:
#           print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#       else:
#           torch.cuda.manual_seed(args.seed)
#   random.seed(args.seed)    
#   if args.pred == 'datetime':
#     pred_datetime()
#   elif args.pred == 'month':
#     pred_month()
#   elif args.pred == 'day':
#     pred_day()
     
#   if args.pred == 'year':
#     pred_year_reg()
#   elif args.pred == 'month':
#     pred_month_reg()
#   elif args.pred == 'day':
#     pred_day_reg()
#   elif args.pred == 'hour':
#     pred_hour_reg()
#   elif args.pred == 'minute':
#     pred_minute_reg()