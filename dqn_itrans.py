import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Importing DataLoaders for each model. These models include rule-based, vanilla DQN and encoder-decoder DQN.
from DataLoader.DataLoader import YahooFinanceDataLoader
from DataLoader.DataForPatternBasedAgent import DataForPatternBasedAgent
from DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from DataLoader.DataSequential import DataSequential 

from DeepRLAgent.MLPEncoder.Train import Train as SimpleMLP
from DeepRLAgent.SimpleCNNEncoder.Train import Train as SimpleCNN
from EncoderDecoderAgent.GRU.Train import Train as gru
from EncoderDecoderAgent.CNN.Train import Train as cnn
from EncoderDecoderAgent.CNN2D.Train import Train as cnn2d
from EncoderDecoderAgent.CNNAttn.Train import Train as cnn_attn
from EncoderDecoderAgent.CNN_GRU.Train import Train as cnn_gru


# Imports for Deep RL Agent
from DeepRLAgent.VanillaInput.Train import Train as DeepRL



# Imports for RL Agent with n-step SARSA
# from RLAgent.Train import Train as RLTrain

# Imports for Rule-Based
from PatternDetectionInCandleStick.LabelPatterns import label_candles
from PatternDetectionInCandleStick.Evaluation import Evaluation

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import matplotlib.pyplot as plt
import seaborn as sns
# from kaleido.scopes.plotly import PlotlyScope
# import plotly.graph_objs as go
import numpy as np
import pandas as pd
import os
import random

from EncoderDecoderAgent.iTransformer.Train import Train as iTransformer
import argparse

CURRENT_PATH = os.getcwd()
BATCH_SIZE = 10
GAMMA=0.7
n_step = 10
BATCH_SIZE = 10
EPS = 0.1
ReplayMemorySize = 20
TARGET_UPDATE = 5
# window_size = 20
num_episodes = 3
n_classes = 64
hidden_size = 16
initial_investment = 1000
# Agent with Auto pattern extraction
# State Mode
state_mode = 5  # OHLC
# state_mode = 2  # OHLC + trend
# state_mode = 3  # OHLC + trend + %body + %upper-shadow + %lower-shadow
window_size = 15

#-------------------------------------------------------------------------------------------------------
train_portfolios = {}
test_portfolios = {}
window_size_experiment = {}
# window_sizes = [3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75]

def add_train_portfo(model_name, portfo):
    counter = 0
    key = f'{model_name}'
    while key in train_portfolios.keys():
        counter += 1
        key = f'{model_name}{counter}'
        
    train_portfolios[key] = portfo

def add_test_portfo(model_name, portfo):
    counter = 0
    key = f'{model_name}'
    while key in test_portfolios.keys():
        counter += 1
        key = f'{model_name}{counter}'
    
    test_portfolios[key] = portfo
#----------------------------------------------------------------------------------------------------------------------
# GOOGL

DATASET_NAME = 'GOOGL'
DATASET_FOLDER = 'GOOGL'
FILE = 'GOOGL.csv'
data_loader = YahooFinanceDataLoader(DATASET_FOLDER, '2018-01-01', load_from_file=True)
transaction_cost = 0.0


dataTrain_autoPatternExtractionAgent = DataAutoPatternExtractionAgent(data_loader.data_train, state_mode, 'action_encoder_decoder', device, GAMMA, n_step, BATCH_SIZE, window_size, transaction_cost)
dataTest_autoPatternExtractionAgent = DataAutoPatternExtractionAgent(data_loader.data_test, state_mode, 'action_encoder_decoder', device, GAMMA, n_step, BATCH_SIZE, window_size, transaction_cost)
# dataTrain_patternBased = DataForPatternBasedAgent(data_loader.data_train, data_loader.patterns, 'action_deepRL', device, GAMMA, n_step, BATCH_SIZE, transaction_cost)
# dataTest_patternBased = DataForPatternBasedAgent(data_loader.data_test, data_loader.patterns, 'action_deepRL', device, GAMMA, n_step, BATCH_SIZE, transaction_cost)

dataTrain_sequential = DataSequential(data_loader.data_train,
                           'action_encoder_decoder', device, GAMMA,
                           n_step, BATCH_SIZE, window_size, transaction_cost)
dataTest_sequential = DataSequential(data_loader.data_test,
                          'action_encoder_decoder', device, GAMMA,
                          n_step, BATCH_SIZE, window_size, transaction_cost)  
#------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='iTransformer')
# # basic config
    # parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    # parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    # parser.add_argument('--model', type=str, required=True, default='iTransformer',
    #                     help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')
parser.add_argument('--is_training', type=int,  default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str,  default='iTransformer',
                    help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')

# data loader
# parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
parser.add_argument('--data', type=str, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=15, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
parser.add_argument('--d_model', type=int, default=128, help='dimension of model') # token维度
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

# iTransformer
parser.add_argument('--exp_name', type=str, required=False, default='None',
                    help='experiemnt name, options:[partial_train, zero_shot]')
parser.add_argument('--efficient_training', type=bool, default=False, help='whether to use efficient_training (exp_name should be partial train)')
parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')

args = parser.parse_args()
#-------------------------------------------------------------------------------------------------
n_classes = 64

simpleMLP = SimpleMLP(data_loader, dataTrain_autoPatternExtractionAgent, dataTest_autoPatternExtractionAgent, DATASET_NAME,
                        state_mode, window_size, transaction_cost, n_classes, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, 
                        ReplayMemorySize=ReplayMemorySize, TARGET_UPDATE=TARGET_UPDATE, n_step=n_step)
# ↓原来的
# simpleMLP = SimpleMLP(data_loader, dataTrain_patternBased, dataTest_patternBased, DATASET_NAME, 
#                     state_mode, window_size, transaction_cost, n_classes, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, EPS=EPS,
#                     ReplayMemorySize=ReplayMemorySize, TARGET_UPDATE=TARGET_UPDATE, n_actions=n_actions, n_step=n_step)

simpleMLP.train(num_episodes)
# -------------------------------------------------------------------------------------------------------------------------------
# itrans = iTransformer(data_loader, dataTrain_autoPatternExtractionAgent, dataTest_autoPatternExtractionAgent, DATASET_NAME, transaction_cost,
#                     config=args,
#                     hidden_size=hidden_size,
#                     BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA,
#                     ReplayMemorySize=ReplayMemorySize,
#                     TARGET_UPDATE=TARGET_UPDATE,
#                     n_step=n_step,
#                     window_size=window_size)


# # ↓原来的
# # simpleMLP = SimpleMLP(data_loader, dataTrain_patternBased, dataTest_patternBased, DATASET_NAME, 
# #                     state_mode, window_size, transaction_cost, n_classes, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, EPS=EPS,
# #                     ReplayMemorySize=ReplayMemorySize, TARGET_UPDATE=TARGET_UPDATE, n_actions=n_actions, n_step=n_step)

# itrans.train(num_episodes)
# file_name = None

# file_name = 'GOOGL; MLP; StateMode(1); WindowSize(20); TRAIN_TEST_SPLIT(True); BATCH_SIZE10; GAMMA0.7; EPSILON0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10; EXPERIMENT.pkl'
# # file_name = 'AAPL; MLP; StateMode(1); WindowSize(20); TRAIN_TEST_SPLIT(True); BATCH_SIZE10; GAMMA0.7; EPSILON0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10; EXPERIMENT.pkl'
# # file_name = 'KSS; DATA_KIND(AutoPatternExtraction); BEGIN_DATE(None); END_DATE(None); SPLIT_POINT(2018-01-01); MLP; TC(0.0); StateMode(1); WindowSize(3); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10(1).pkl'
# # file_name = 'BTC-USD; MLP; StateMode(1); WindowSize(20); TRAIN_TEST_SPLIT(True); BATCH_SIZE10; GAMMA0.7; EPSILON0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10; EXPERIMENT.pkl'
# # file_name = 'GE; DATA_KIND(AutoPatternExtraction); BEGIN_DATE(None); END_DATE(None); SPLIT_POINT(2015-01-01); MLP; TC(0); StateMode(1); WindowSize(20); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10.pkl'
# # file_name = 'HSI; DATA_KIND(AutoPatternExtraction); BEGIN_DATE(None); END_DATE(None); SPLIT_POINT(2015-01-01); MLP; TC(0); StateMode(1); WindowSize(20); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10.pkl'
# # file_name = 'AAL; DATA_KIND(AutoPatternExtraction); BEGIN_DATE(None); END_DATE(None); SPLIT_POINT(2018-01-01); MLP; TC(0); StateMode(1); WindowSize(15); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10.pkl'

# ev_iTransformer_train = itrans.test(
#                                   initial_investment=initial_investment, test_type='train')

# iTransformer_portfolio_train = ev_iTransformer_train.get_daily_portfolio_value()

# ev_iTransformer_test = itrans.test(
#                                   initial_investment=initial_investment, test_type='test')
# iTransformer_portfolio_test = ev_iTransformer_test.get_daily_portfolio_value()

# model_kind = 'iTransformer'

# add_train_portfo(model_kind, iTransformer_portfolio_train)
# add_test_portfo(model_kind, iTransformer_portfolio_test)

# -----------------------------------------------------------------------------



# -------------------------------------------------------
experiment_num = 1
RESULTS_PATH = 'TestResults/Train/'

import os

if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)

while os.path.exists(f'{RESULTS_PATH}{DATASET_NAME};train;EXPERIMENT({experiment_num}).jpg'):
    experiment_num += 1

fig_file = f'{RESULTS_PATH}{DATASET_NAME};train;EXPERIMENT({experiment_num}).jpg'

sns.set(rc={'figure.figsize': (15, 7)})

items = list(test_portfolios.keys())
random.shuffle(items)

first = True
for k in items:
    profit_percentage = [(train_portfolios[k][i] - train_portfolios[k][0])/train_portfolios[k][0] * 100 
              for i in range(len(train_portfolios[k]))]
    difference = len(train_portfolios[k]) - len(data_loader.data_train_with_date)
    df = pd.DataFrame({'date': data_loader.data_train_with_date.index, 
                       'portfolio':profit_percentage[difference:]})
    if not first:
        df.plot(ax=ax, x='date', y='portfolio', label=k)
    else:
        ax = df.plot(x='date', y='portfolio', label=k)
        first = False

ax.set(xlabel='Time', ylabel='%Rate of Return')
ax.set_title(f'%Rate of Return at each point of time for training data of {DATASET_NAME}')
        
plt.legend()
plt.savefig(fig_file, dpi=300)

#----------------------------------------------------------------------
shuffle = True
import random

experiment_num = 1
RESULTS_PATH = 'TestResults/Test/'

import os

if not os.path.exists(RESULTS_PATH):
    os.mkdir(RESULTS_PATH)

while os.path.exists(f'{RESULTS_PATH}{DATASET_NAME};test;EXPERIMENT({experiment_num}).jpg'):
    experiment_num += 1

fig_file = f'{RESULTS_PATH}{DATASET_NAME};test;EXPERIMENT({experiment_num}).jpg'

sns.set(rc={'figure.figsize': (15, 7)})
sns.set_palette(sns.color_palette("Paired", 15))

items = list(test_portfolios.keys())

if shuffle:
    random.shuffle(items)

first = True
for k in items:
    profit_percentage = [(test_portfolios[k][i] - test_portfolios[k][0])/test_portfolios[k][0] * 100 
                  for i in range(len(test_portfolios[k]))]
    difference = len(test_portfolios[k]) - len(data_loader.data_test_with_date)
    df = pd.DataFrame({'date': data_loader.data_test_with_date.index, 
                       'portfolio':profit_percentage[difference:]})
    if not first:
        df.plot(ax=ax, x='date', y='portfolio', label=k)
    else:
        ax = df.plot(x='date', y='portfolio', label=k)
        first = False
        
ax.set(xlabel='Time', ylabel='%Rate of Return')
ax.set_title(f'Comparing the %Rate of Return for different models '
             f'at each point of time for test data of {DATASET_NAME}')
plt.legend()
plt.savefig(fig_file, dpi=300)
