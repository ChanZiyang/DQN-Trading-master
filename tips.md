data[action]的获取写在了PatternDetectionInCandleStick\LabelPatterns.py里，在YahooFinanceDataLoader初始化就会调用

训练器的test()添加了模型预测的action在data里送入了evaluation

外置的add_train_portfo（dict）记载了每个算法的资产时间序列

DataLoader\DataAutoPatternExtractionAgent.py 是基于深度强化学习的
DataLoader\DataForPatternBasedAgent.py 是基于传统pattern判断的
DataLoader\DataSequential 是序列模型

DataLoader.data的 mean_candle 列就是close

train的时候，每次push一个item进去replaybuffer，直到达到BATCH_SIZE就开始训练，再每次抽取BATCH_SIZE个item出来训练

对比学习：
DataLoader\DataLoader.py 里面加了reward_label
DeepRLAgent\BaseTrain.py 里面加了Supcon的训练细节

next_state 的 第56个为None