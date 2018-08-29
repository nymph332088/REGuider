from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.functional as F
import torch.nn as nn
import os

class BiLSTMTagger(nn.Module):
    def __init__(self, config):
        super(BiLSTMTagger, self).__init__()
        self.bilstm = BiLSTM(config)
    
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(config['nhid'] * 2, config['target_size'])
        
      
      
    def init_weights(self, init_range=0.1):
        self.hidden2tag.weight.data.uniform_(-init_range, init_range)
        self.hidden2tag.bias.data.fill_(0)

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)
      
    def forward(self, inp, hidden):
        # inp size: [len, bsz] tokens
        # hidden size: [nlayers * 2, bsz, nhidden]
        # outp size: [bsz, len, nhid*2]
        
        outp, emb = self.bilstm.forward(inp, hidden)
        size = outp.size()  # [bsz, len, nhid*2]
        # print(size)
        new_outp = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        
        tag_space = self.hidden2tag(new_outp) # [bsz*len, vocab size]
        # tag_scores = F.log_softmax(tag_space, dim=1).view(size[0], size[1], -1) # [bsz, len, vocab size]
        tag_scores = nn.LogSoftmax(dim=1)(tag_space).view(size[0], size[1], -1)
        return tag_scores, outp
      

    def encode(self, inp, hidden):
        return self.forward(inp, hidden)[0]

      
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.FILTERS = config["filters"]
        self.FILTER_NUM = config["filter_nums"]
        self.DROPOUT_PROB = config["dropout"]
        self.IN_CHANNEL = config['ninp']
    
        assert (len(self.FILTERS) == len(self.FILTER_NUM))
    
        # one for zero padding
        self.embedding = nn.Embedding(config['ntoken'], config['ninp'], padding_idx=0)
        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.FILTERS[i])
            setattr(self, 'conv_%s'%(i), conv)

        self.relu = nn.ReLU()
    def get_conv(self, i):
        return getattr(self, 'conv_%s'%(i)) 
  
    def init_hidden(self, bsz):
        return None
      
    def forward(self, inp, hidden):
        # size of inp: [len, bsz] for the sake of LSTM needs [len, bsz, emb_size]
        #[len, bsz, emb_size] => [bsz, emb_size, len]
        emb = self.embedding(inp).permute(1, 2, 0)
          
        conv_results = []
        for i in range(len(self.FILTERS)):
          # [bsz, num_filter_i, len_out]
          conv_r = self.get_conv(i)(emb)
          conv_r = self.relu(conv_r)
          size = conv_r.size()
          # [bsz, num_filter_i, 1]
          pool = nn.MaxPool1d(size[2])(conv_r)
          # [bsz, num_filter_i]
          conv_results.append(pool.view(size[0], -1))
        # [ bsz, num_filter_total]
        outp = torch.cat(conv_results, 1)
        return outp, emb
    
class MLP(nn.Module):
  def __init__(self, config):
    super(MLP, self).__init__()
    self.embedding = nn.Embedding(config['ntoken'], config['ninp'], padding_idx=0)

  def init_hidden(self, bsz):
    return None
  
  def forward(self, inp, hidden):
    # [len, bsz] => [len, bsz, emb_dim] => [bsz, len, emb_dim]
    emb = self.embedding(inp).permute(1, 0, 2).contiguous()
    size = emb.size()
    outp = emb.view(size[0], -1)
    return outp, None
    
class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.encoder = nn.Embedding(config['ntoken'], config['ninp'])
        self.bilstm = nn.LSTM(config['ninp'], config['nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=True)
        self.nlayers = config['nlayers']
        self.nhid = config['nhid']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']
#        self.init_weights()
        self.encoder.weight.data[self.dictionary.word2idx['<pad>']] = 0
        if os.path.exists(config['word-vector']):
            print('Loading word vectors from', config['word-vector'])
            vectors = torch.load(config['word-vector'])
            assert vectors[2] >= config['ninp']
            vocab = vectors[0]
            vectors = vectors[1]
            loaded_cnt = 0
            for word in self.dictionary.word2idx:
                if word not in vocab:
                    continue
                real_id = self.dictionary.word2idx[word]
                loaded_id = vocab[word]
                self.encoder.weight.data[real_id] = vectors[loaded_id][:config['ninp']]
                loaded_cnt += 1
            print('%d words from external word vectors loaded.' % loaded_cnt)

    # note: init_range constraints the value of initial weights
    def init_weights(self, init_range=0.1):
        self.encoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        # emb size = [len, bsz, emb_size]
        emb = self.drop(self.encoder(inp))
        # print(emb)
        # print(hidden)
        # print(self.bilstm(emb, hidden))
        # outp size = [len, bsz, emb_size]        
        outp = self.bilstm(emb, hidden)[0]
        
        if self.pooling == 'mean':
            outp = torch.mean(outp, 0).squeeze()
        elif self.pooling == 'max':
            outp = torch.max(outp, 0)[0].squeeze()
        elif self.pooling == 'all' or self.pooling == 'all-word':
            outp = torch.transpose(outp, 0, 1).contiguous()
        return outp, emb

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()),
                Variable(weight.new(self.nlayers * 2, bsz, self.nhid).zero_()))


class SelfAttentiveEncoder(nn.Module):

    def __init__(self, config):
        super(SelfAttentiveEncoder, self).__init__()
        self.bilstm = BiLSTM(config)
        self.drop = nn.Dropout(config['dropout'])
        self.ws1 = nn.Linear(config['nhid'] * 2, config['attention-unit'], bias=False)
        self.ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dictionary = config['dictionary']
#        self.init_weights()
        self.attention_hops = config['attention-hops']

    def init_weights(self, init_range=0.1):
        self.ws1.weight.data.uniform_(-init_range, init_range)
        self.ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, inp, hidden):
        outp = self.bilstm.forward(inp, hidden)[0]
        size = outp.size()  # [bsz, len, nhid*2]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == self.dictionary.word2idx['<pad>']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)


class Classifier(nn.Module):

  def __init__(self, config):
      super(Classifier, self).__init__()
      if config['pooling'] == 'mean' or config['pooling'] == 'max':
          self.encoder = BiLSTM(config)
          self.fc = nn.Linear(config['nhid'] * 2, config['nfc'])
          self.mode = 'lstm'
      elif config['pooling'] == 'all':
          self.encoder = SelfAttentiveEncoder(config)
          self.fc = nn.Linear(config['nhid'] * 2 * config['attention-hops'], config['nfc'])
          self.mode = 'alstm'
      elif config['pooling'] == 'conv':
          self.encoder = CNN(config)
          self.fc = nn.Linear(sum(config['filter_nums']), config['nfc'])
          self.mode = 'cnn'
      elif config['pooling'] == 'mlp':
          self.encoder = MLP(config)
          self.fc = nn.Linear(config['max_len']*config['ninp'], config['nfc'])
          self.mode = 'mlp'
      else:
          raise Exception('Error when initializing Classifier')
      self.drop = nn.Dropout(config['dropout'])
      self.tanh = nn.Tanh()
      self.pred = nn.Linear(config['nfc'], config['class-number'])
      self.dictionary = config['dictionary']
      
#        self.init_weights()

  def init_weights(self, init_range=0.1):
      self.fc.weight.data.uniform_(-init_range, init_range)
      self.fc.bias.data.fill_(0)
      self.pred.weight.data.uniform_(-init_range, init_range)
      self.pred.bias.data.fill_(0)

  def forward(self, inp, hidden):
      outp, attention = self.encoder.forward(inp, hidden)
      outp = outp.view(outp.size(0), -1)
      fc = self.tanh(self.fc(self.drop(outp)))
      pred = self.pred(self.drop(fc))
      if type(self.encoder) != SelfAttentiveEncoder:
          attention = None
      return pred, attention

  def init_hidden(self, bsz):
      return self.encoder.init_hidden(bsz)

  def encode(self, inp, hidden):
      return self.encoder.forward(inp, hidden)[0]

        
class MultiTaskClassifier(nn.Module):
  def __init__(self, config):
    super(Classifier, self).__init__()
    if config['pooling'] == 'mean' or config['pooling'] == 'max':
        self.encoder = BiLSTM(config)
        self.fc = nn.Linear(config['nhid'] * 2, config['nfc'])
        self.mode = 'lstm'
    elif config['pooling'] == 'all':
        self.encoder = SelfAttentiveEncoder(config)
        self.fc = nn.Linear(config['nhid'] * 2 * config['attention-hops'], config['nfc'])
        self.mode = 'alstm'
    elif config['pooling'] == 'conv':
        self.encoder = CNN(config)
        self.fc = nn.Linear(sum(config['filter_nums']), config['nfc'])
        self.mode = 'cnn'
    elif config['pooling'] == 'mlp':
        self.encoder = MLP(config)
        self.fc = nn.Linear(config['max_len']*config['ninp'], config['nfc'])
        self.mode = 'mlp'
    else:
        raise Exception('Error when initializing Classifier')
    self.drop = nn.Dropout(config['dropout'])
    self.tanh = nn.Tanh()
    self.preds = [nn.Linear(config['nfc'], class_num) \
                  for class_num in config['class-number']]
    self.dictionary = config['dictionary']
      
    self.init_weights()

  def init_weights(self, init_range=0.1):
    self.fc.weight.data.uniform_(-init_range, init_range)
    self.fc.bias.data.fill_(0)
    for pred in self.preds:
      pred.weight.data.uniform_(-init_range, init_range)
      pred.bias.data.fill_(0)

  def forward(self, inp, hidden):
    outp, attention = self.encoder.forward(inp, hidden)
    outp = outp.view(outp.size(0), -1)
    fc = self.tanh(self.fc(self.drop(outp)))
    pred = self.pred(self.drop(fc))
    if type(self.encoder) != SelfAttentiveEncoder:
      attention = None
    return pred, attention

  def init_hidden(self, bsz):
    return self.encoder.init_hidden(bsz)

  def encode(self, inp, hidden):
    return self.encoder.forward(inp, hidden)[0]
  