import numpy as np
import torch
import os
from torch.utils.data import Dataset
import MetaTrader5 as mt5

class MT5Dataset(Dataset):
    """
    Args:
        path (str) : File storage location
    Variables:
        MT5Dataset.data (tensor) : Data with shape (count, symbols, columns)
    """
    def __init__(self, path=None):
        self.data      = None
        self.date_from = None
        self.data_to   = None
        self.count     = None
        if path:
            self.load(path)

    def standard(self):
        '''Standardize data'''
        self.data -= self.data.mean(0).unsqueeze(0)
        std        = self.data.std(0).unsqueeze(0)
        std[std < 10E-7] = np.inf
        self.data /= std

    def save(self, root=os.getcwd()):
        filename  = str(self.date_from)[:16] + ', '
        filename += str(self.date_to)[:16] + ', '
        filename += str(count) + '.pkl'
        torch.save({
                    'data': self.data,
                    'date_from': self.date_from,
                    'date_to': self.date_to,
                    'count': self.count
                    }, root + filename)

    def load(self, path):
        datainfo = torch.load(path)
        self.data      = datainfo['data']
        self.date_from = datainfo['date_from']
        self.data_to   = datainfo['data_to']
        self.count     = datainfo['count']

    def copy(self, symbol, date_from, date_to=None, count=None, timeframe=mt5.TIMEFRAME_H1):
        """
        Args:
            symbol    (list)    : List of symbols
                Get all symbols : [i.name for i in mt5.symbols_get()]
            date_from (datetime): Date, the ticks are requested from
            date_to   (datetime): Date, up to which the ticks are requested
            count     (int)     : The length of the data (minutes)
            timeframe (int)     : Time timeframe of data sampling (mt5.TIMEFRAME)
                Form : mt5.TIMEFRAME_ + M(minute) / H(hour) / D(day) / W(week) / MN(month) + value
        """
        self.data = []
        self.date_from = date_from
        self.data_to   = date_to
        self.count     = count
        for i in range(len(symbol)):
            if count:
                data = mt5.copy_rates_from(symbol[i], timeframe, date_from, count).tolist()
            elif date_to:
                data = mt5.copy_rates_range(symbol[i], timeframe, date_from, date_to).tolist()
            else:
                print('Argument passed in error')
                return

            if data is None:
                print(f'{symbol[i]} data copy error, skip')
            else:
                self.data += [data]

        self.data = torch.FloatTensor(self.data).permute(1, 0, 2)

    def __getitem__(self, X):
        '''
            You can also use MT5Dataset.data[X] directly
        '''
        return self.data[X]

class DataLoader:
    def __init__(self, dataset, seq_len, batch_size=1, label_len=1, 
                    overlap=True, shuffle=True, batch_first=False, unique=False):
        """
        The last incomplete batch will be smaller
        Args:
            dataset (MT5Dataset) : MT5Dataset
            seq_len     (int)    : The length of the sequence
            batch_size  (int)    : How many samples per batch to load
            label_len   (int)    : Label's time step
            overlap     (bool)   : Whether the sample sequence is disjoint
            shuffle     (bool)   : Set to True to have the data reshuffled
            batch_first (bool)   : Whether to use batch as dim=0
            unique      (bool)   : Analysis time between unsynchronized symbols
        """
        self.idx = 0
        self.dataset     = dataset
        self.seq_len     = seq_len
        self.batch_size  = batch_size
        self.batch_first = batch_first
        self.unique      = unique
        shape = dataset.data.shape
        self.Y = np.arange(shape[1])

        if overlap:
            step = seq_len
        else:
            step = 1

        self.index = np.arange(stop=shape[0]-seq_len+1, step=step)
        self.index = np.stack([self.index] * shape[1])
        self.index = np.stack([self.index+i for i in range(seq_len)])
        self.index = self.index.transpose(0, 2, 1)
        # index shape : (seq_len, batch_num, symbols)
        self.batch_num = self.index.shape[1]
        
        if shuffle:
            self.shuffle()
        # train, test

    def shuffle(self):
        if self.unique:
            for i in self.index[0].T:
                np.random.default_rng().shuffle(i)
        else:
            np.random.default_rng().shuffle(self.index[0])
        for i in range(1, self.seq_len):
            self.index[i] = self.index[i-1]+1

    def __len__(self):
        return self.batch_num
        
    def __iter__(self):
        self.idx = 0
        return self       
        
    def __next__(self):
        if self.idx < self.batch_num:
            X = self.index[:, self.idx*self.batch_size:(self.idx+1)*self.batch_size]
            # X shape : (seq_len, batch_size, symbols)
            inputs = self.dataset.data[X, self.Y]
            # inputs shape : (seq_len, batch_size, symbols, columns)

            if self.batch_first:
                return inputs.transpose(1, 0, 2, 3)

            return inputs
        else:
            raise StopIteration


# example
if __name__ == '__main__':
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
    timeframe = mt5.TIMEFRAME_M30

    demo_set = MT5Dataset()
    demo_set.copy(symbol, time1, time2, timeframe=timeframe)
    demo_set.standard()
    print('data[:2, 0] : \n', demo_set.data[:2, 0])
    print('shape of data (count, symbols, columns) : ', demo_set.data.shape)


# class a():
#     def __init__(self):
#         self.data = torch.randint(0, 10, [11,7,5])

# dataset = a()
# shape = dataset.data.shape  # (count, symbols, columns)
# batch_size = 3
# seq_len = 2
# idx = 0
# batch_num = (shape[0]-seq_len+1) // batch_size
# index = np.stack([np.arange(shape[0]-seq_len+1)] * shape[1])
# index = np.stack([index+i for i in range(seq_len)])
# index = index.transpose(0, 2, 1)
# for i in index[0]:
#         np.random.default_rng().shuffle(index[0])

# for k in range(1, seq_len):
#             index[k] = index[k-1]+1

# X = index[:, idx:idx+batch_size]

# # data  : (count, symbols, columns) (11, 7, 5)
# # index : (seq_len, count-seq_len+1, symbols, ) (2, 10, 7)

# idx = 3
# dataset.data[X[0, idx:idx + batch_size]]