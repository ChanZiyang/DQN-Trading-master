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
from DeepRLAgent.MLPEncoder.Train_SC import Train as SCMLP
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


CURRENT_PATH = os.getcwd()
BATCH_SIZE = 64
GAMMA=0.7
n_step = 10
EPS = 0.1
ReplayMemorySize = 256
TARGET_UPDATE = 5
# window_size = 20
num_episodes = 200
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



#-----------------------------------------------------------------------------------------------------------
dataTrain_autoPatternExtractionAgent_SC = DataAutoPatternExtractionAgent(data_loader.data_train, state_mode, 'action_encoder_decoder', device, GAMMA, n_step, BATCH_SIZE, window_size, transaction_cost,is_SC=True)
dataTest_autoPatternExtractionAgent_SC = DataAutoPatternExtractionAgent(data_loader.data_test, state_mode, 'action_encoder_decoder', device, GAMMA, n_step, BATCH_SIZE, window_size, transaction_cost,is_SC=True)

scMLP = SCMLP(data_loader, dataTrain_autoPatternExtractionAgent_SC, dataTest_autoPatternExtractionAgent_SC, DATASET_NAME,
                        state_mode, window_size, transaction_cost, n_classes, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, 
                        ReplayMemorySize=ReplayMemorySize, TARGET_UPDATE=TARGET_UPDATE, n_step=n_step)


scMLP.train(num_episodes)

ev_scMLP_train = scMLP.test(
                                  initial_investment=initial_investment, test_type='train')
scMLP_portfolio_train = ev_scMLP_train.get_daily_portfolio_value()
ev_scMLP_test = scMLP.test(
                                  initial_investment=initial_investment, test_type='test')
scMLP_portfolio_test = ev_scMLP_test.get_daily_portfolio_value()

model_kind = 'MLP-Supcon'

add_train_portfo(model_kind, scMLP_portfolio_train)
add_test_portfo(model_kind, scMLP_portfolio_test)

#----------------------------------------------------------------------------
n_classes = 64
dataTrain_autoPatternExtractionAgent = DataAutoPatternExtractionAgent(data_loader.data_train, state_mode, 'action_encoder_decoder', device, GAMMA, n_step, BATCH_SIZE, window_size, transaction_cost)
dataTest_autoPatternExtractionAgent = DataAutoPatternExtractionAgent(data_loader.data_test, state_mode, 'action_encoder_decoder', device, GAMMA, n_step, BATCH_SIZE, window_size, transaction_cost)

simpleMLP = SimpleMLP(data_loader, dataTrain_autoPatternExtractionAgent, dataTest_autoPatternExtractionAgent, DATASET_NAME,
                        state_mode, window_size, transaction_cost, n_classes, BATCH_SIZE=BATCH_SIZE, GAMMA=GAMMA, 
                        ReplayMemorySize=ReplayMemorySize, TARGET_UPDATE=TARGET_UPDATE, n_step=n_step)


simpleMLP.train(num_episodes)

ev_simpleMLP_train = simpleMLP.test(
                                  initial_investment=initial_investment, test_type='train')
simpleMLP_portfolio_train = ev_simpleMLP_train.get_daily_portfolio_value()
ev_simpleMLP_test = simpleMLP.test(
                                  initial_investment=initial_investment, test_type='test')
simpleMLP_portfolio_test = ev_simpleMLP_test.get_daily_portfolio_value()

model_kind = 'MLP-vanilla'

add_train_portfo(model_kind, simpleMLP_portfolio_train)
add_test_portfo(model_kind, simpleMLP_portfolio_test)

#-------------------
print('train:',train_portfolios['MLP-vanilla'][-1],train_portfolios['MLP-Supcon'][-1])
print('test:',test_portfolios['MLP-vanilla'][-1],test_portfolios['MLP-Supcon'][-1])

RESULTS_PATH = 'TestResults/Test/'
fig_file = f'{RESULTS_PATH}{DATASET_NAME};test;EXPERIMENT(new).jpg'

items = list(test_portfolios.keys())

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

RESULTS_PATH = 'TestResults/Train/'
fig_file = f'{RESULTS_PATH}{DATASET_NAME};train;EXPERIMENT(new).jpg'


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