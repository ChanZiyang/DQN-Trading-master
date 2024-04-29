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
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import os
import random

CURRENT_PATH = os.getcwd()
BATCH_SIZE = 10
GAMMA=0.7
n_step = 10

initial_investment = 1000


train_portfolios = {}
test_portfolios = {}
window_size_experiment = {}
window_sizes = [3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75]


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


# GOOGL
DATASET_NAME = 'GOOGL'
DATASET_FOLDER = 'GOOGL'
FILE = 'GOOGL.csv'
data_loader = YahooFinanceDataLoader(DATASET_FOLDER, '2018-01-01', load_from_file=True)
transaction_cost = 0.0
# Agent with Auto pattern extraction
# State Mode
state_mode = 1  # OHLC
# state_mode = 2  # OHLC + trend
# state_mode = 3  # OHLC + trend + %body + %upper-shadow + %lower-shadow

window_size = 15
dataTrain_sequential = DataSequential(data_loader.data_train,
                           'action_encoder_decoder', device, GAMMA,
                           n_step, BATCH_SIZE, window_size, transaction_cost)
dataTest_sequential = DataSequential(data_loader.data_test,
                          'action_encoder_decoder', device, GAMMA,
                          n_step, BATCH_SIZE, window_size, transaction_cost)  

dataTrain_autoPatternExtractionAgent = DataAutoPatternExtractionAgent(data_loader.data_train, state_mode, 'action_encoder_decoder', device, GAMMA, n_step, BATCH_SIZE, window_size, transaction_cost)
dataTest_autoPatternExtractionAgent = DataAutoPatternExtractionAgent(data_loader.data_test, state_mode, 'action_encoder_decoder', device, GAMMA, n_step, BATCH_SIZE, window_size, transaction_cost)
dataTrain_patternBased = DataForPatternBasedAgent(data_loader.data_train, data_loader.patterns, 'action_deepRL', device, GAMMA, n_step, BATCH_SIZE, transaction_cost)
dataTest_patternBased = DataForPatternBasedAgent(data_loader.data_test, data_loader.patterns, 'action_deepRL', device, GAMMA, n_step, BATCH_SIZE, transaction_cost)


BATCH_SIZE = 10
EPS = 0.1
# EPS_START = 0.9
# EPS_END = 0.05
# EPS_DECAY = 200

ReplayMemorySize = 20

TARGET_UPDATE = 5
n_actions = 3
# window_size = 20

num_episodes = 3
n_classes = 64

# -------------------------------------------------------------------------------------------------------------------------------
hidden_size = 4

gru_agent = gru(data_loader, dataTrain_sequential, dataTest_sequential, DATASET_NAME, transaction_cost, hidden_size,
                    BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA,
                    ReplayMemorySize=ReplayMemorySize,
                    TARGET_UPDATE=TARGET_UPDATE,
                    n_step=n_step,
                    window_size=window_size)

gru_agent.train(num_episodes)
file_name = None

file_name = 'GOOGL; DATA_KIND(LSTMSequential); GRU; PredictionStep(None); WindowSize(20); TRAIN_TEST_SPLIT(True); BATCH_SIZE10; GAMMA0.7; EPSILON0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10; EXPERIMENT.pkl'
# file_name = 'AAPL; DATA_KIND(LSTMSequential); GRU; PredictionStep(None); WindowSize(20); TRAIN_TEST_SPLIT(True); BATCH_SIZE10; GAMMA0.7; EPSILON0.7; REPLAY_MEMORY_SIZE20; C5; N_SARSA10; EXPERIMENT.pkl'
# file_name = 'BTC-USD; DATA_KIND(LSTMSequential); GRU; PredictionStep(None); WindowSize(20); TRAIN_TEST_SPLIT(True); BATCH_SIZE10; GAMMA0.7; EPSILON0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10; EXPERIMENT.pkl'
# file_name = 'KSS; DATA_KIND(LSTMSequential); Dates(None, 2018-01-01, None); GRU; TC(0.0); WindowSize(20); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10.pkl'
# file_name = 'GE; DATA_KIND(LSTMSequential); Dates(None, 2015-01-01, None); GRU; TC(0); WindowSize(15); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10.pkl'
# file_name = 'HSI; DATA_KIND(LSTMSequential); Dates(None, 2015-01-01, None); GRU; TC(0); WindowSize(15); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10.pkl'
# file_name = 'AAL; DATA_KIND(LSTMSequential); Dates(None, 2018-01-01, None); GRU; TC(0); WindowSize(10); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10.pkl'

# file_name = 'BTC-USD; DATA_KIND(LSTMSequential); Dates(None, 2018-01-01, None); GRU; TC(0); WindowSize(3); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10(1).pkl'
# file_name = 'GOOGL; DATA_KIND(LSTMSequential); Dates(None, 2018-01-01, None); GRU; TC(0); WindowSize(3); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10(1).pkl'
# file_name = 'AAPL; DATA_KIND(LSTMSequential); Dates(2010-01-01, 2018-01-01, 2020-08-24); GRU; TC(0); WindowSize(3); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10(1).pkl'
# file_name = 'KSS; DATA_KIND(LSTMSequential); Dates(None, 2018-01-01, None); GRU; TC(0.0); WindowSize(3); BATCH_SIZE10; GAMMA0.7; EPS0.1; REPLAY_MEMORY_SIZE20; C5; N_SARSA10(1).pkl'

model_kind = 'GRU'

ev_gru_agent = gru_agent.test(file_name=file_name, action_name=dataTrain_sequential.action_name,
                                  initial_investment=initial_investment, test_type='train')
gru_agent_portfolio_train = ev_gru_agent.get_daily_portfolio_value()
ev_gru_agent = gru_agent.test(file_name=file_name, action_name=dataTrain_sequential.action_name,
                                  initial_investment=initial_investment, test_type='test')
gru_agent_portfolio_test = ev_gru_agent.get_daily_portfolio_value()

model_kind = 'GRU'

add_train_portfo('GRU', gru_agent_portfolio_train)
add_test_portfo('GRU', gru_agent_portfolio_test)

