# This script is for training models
from dataloader import MT5Dataset, DataLoader
from utils import plot_loss
from train import train_model
from models import dumbmodel

from datetime import datetime
import pytz

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()

symbol = [i.name for i in mt5.symbols_get()][:3]
timezone = pytz.timezone("Etc/UTC")
time1 = datetime(2021,1,27,15, tzinfo=timezone)
time2 = datetime(2021,1,28,15, tzinfo=timezone)
count = 10
timeframe = mt5.TIMEFRAME_M20

demo_set = MT5Dataset()
demo_set.copy(symbol, time1, time2, timeframe=timeframe)
demo_set.standard()
print('data[:2, 0] : \n', demo_set.data[:2, 0])
print('shape of data (count, symbols, columns) : ', demo_set.data.shape)

# Dataloader
print('--Dataloader test--')
seq_len=5
batch_size=3
label_len=2
dataloader = DataLoader(demo_set, seq_len, batch_size, label_len)
for i, j in dataloader:
    print('inputs shape : (seq_len, batch_size, symbols, columns)', i.shape)
    print('labels shape : (label_len, batch_size, symbols, columns)', j.shape)
    break

# train model
lr = 1
weight_decay = 0.1
optim = torch.optimizer.Adam(dumbmodel.parameters(), lr=lr, weight_decay=weight_decay)
best_model, stats = train_model(dumbmodel, dataloader, dataloader, optim)
plot_loss(stats)