import MetaTrader5 as mt5
# 文檔 : https://www.mql5.com/zh/docs/integration/python_metatrader5

# 連接到 MetaTrader 5
if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
 
# 請求連接狀態和參數
# print(mt5.terminal_info())
# 獲取有關 MetaTrader 5 版本的數據
# print(mt5.version())
 
# 請求 2020/1/28 13:00 開始的 1000 個 EURAUD 
# 由 MT5 取得的數據為 UTC+0
# 返回  時間(秒),   賣價,     買價,    最後價, 成交量, 時間(毫秒),     flags, volume_real
# 例 : (1580209200, 1.63412, 1.63437, 0.,     0,      1580209200067, 130,   0.)
from datetime import datetime
euraud_ticks = mt5.copy_ticks_from("EURAUD", datetime(2020,1,28,13), 1000, mt5.COPY_TICKS_ALL)
print(len(euraud_ticks))

# 顯示所有交易品種
# for i in mt5.symbols_get():
#     print(i.name)

# 請求 2020/1/27 13:00 - 2020/1/28 13:00 之間的 AUDUSD 所有報價
import pytz
timezone = pytz.timezone("Etc/UTC")
time1 = datetime(2021,1,27,15, tzinfo=timezone)
time2 = datetime(2021,1,28,9, tzinfo=timezone)
audusd_ticks = mt5.copy_ticks_range("AUDUSD", time1, time2, mt5.COPY_TICKS_ALL)
print(len(audusd_ticks))
audusd_ticks = mt5.copy_ticks_range("USDJPY", time1, time2, mt5.COPY_TICKS_ALL)
print(len(audusd_ticks))
 
# 通過多種方式獲取不同交易品種的柱形圖
# Args : 交易品種, 時間週期, 開盤日期, 樣本數目
# 返回  時間,       開盤價,  最高價,   最低價, 收盤價,  tick_volume, 點差, real_volume
# 例 : (1578484800, 1.11382, 1.11385, 1.1111, 1.11199, 9354,        1,    0)
eurusd_rates = mt5.copy_rates_from("EURUSD", mt5.TIMEFRAME_M1, datetime(2020,1,28,13), 1000)
eurgbp_rates = mt5.copy_rates_from_pos("EURGBP", mt5.TIMEFRAME_M1, 0, 1000)
eurcad_rates = mt5.copy_rates_range("EURCAD", mt5.TIMEFRAME_M5, datetime(2020,1,27,13), datetime(2020,1,28,13))
a = mt5.copy_rates_range("EURCAD", mt5.TIMEFRAME_M1, datetime(2020,1,27,13), datetime(2020,1,28,13))
b = mt5.copy_rates_range("EURCAD", mt5.TIMEFRAME_M1, datetime(2020,1,27,13), datetime(2020,1,28,13))
c = mt5.copy_rates_range("EURCAD", mt5.TIMEFRAME_M1, datetime(2020,1,27,13), datetime(2020,1,28,13))
print(len(a), len(b), len(c))
# 與 MetaTrader 5 結束連接
# mt5.shutdown()