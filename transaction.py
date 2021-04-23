import numpy as np

def strategy(sell, buy, tick=None):
    '''
    假設買價和賣價的漲跌勢相同
    input:
        sell  賣價
        buy   買價
        tick  交易時間
    output:
        date  交易點
        price 交易價
    '''
    if type(tick) == type(None):
        tick = np.arange(len(sell))
    # 第一部分 : 判斷所有極值
    # 依買賣價數列的 index 逐個檢查賣價的最大值和買價的最小值
    # 找到極值後，紀錄其值於 price，記錄交易時間於 date

    # 左端點的判斷
    # 若 sell[0] > sell[1]，則視左端點為最大值；反之為最小值
    # 如果第一個找到的是賣價的極值，則令 odd = 1
    date = np.array(tick[0])
    if sell[0] > sell[1]:
        odd = 1
        price = np.array([sell[0]])  # 紀錄賣價的最大值
    else:
        odd = 0
        price = np.array([buy [0]])  # 紀錄買價的最小值

    # 主體的判斷
    # 小於等於是為了處理相鄰兩值相等的問題，不是很重要
    for i in range(1, len(sell)-1):
        if sell[i-1] <= sell[i] >  sell[i+1]:
            price = np.append(price, sell[i])
            date  = np.append(date, tick[i])
        if buy [i-1] >  buy [i] <= buy [i+1]:
            price = np.append(price, buy [i])
            date  = np.append(date, tick[i])

    # 右端點的判斷
    # 若 sell[-2] <= sell[-1]，則視右端點為最大值；反之為最小值
    if sell[-2] <= sell[-1]:
        price = np.append(price, sell[-1])
        date  = np.append(date, tick[-1])
    else:
        price = np.append(price, buy [-1])
        date  = np.append(date, tick[-1])

    # 第二部分 : 最佳化
    # 假設賣價的最大值 s，由左至右的順序為 s1, s2, ...
    # 假設買價的最小值 b，由左至右的順序為 b1, b2, ...
    # 令最大與最小值交錯相鄰，如下述 : 
    #   如果 odd = 1，則 price = s1, b1, s2, b2, ...
    #   如果 odd = 0，則 price = b1, s1, b2, s2, ...
    #   即，若 odd=1，則 price 以 s 開頭，否則以 b 開頭
    # price 內每一對相鄰的元素，皆可形成一筆獲利視為 s-b 的交易
    # 賣價 s 越高越優等，買價 b 越低越優等

    # 左端點的最佳化
    # 假設 odd = 1，則 price[:4] = s1, b1, s2, b2
    # 若第一筆交易虧損 (s1-b1 < 0)，則必須刪除 s1, b1 其中之一
    # 接著找出刪除的可能性內最優等的方法
    # if   s1 < s2，則刪除 s1      (因為 s1 比 s2 劣等)
    #      因為變成 b1 開頭，所以 odd 要轉換
    # elif b2 < b1，則刪除 s2, b1  (因為 b1 比 b2 劣等)
    #      因為 s2 劣等而 s, b 須交錯，所以刪除 s2
    # else        ，則刪除 s2, b2  (因為 b2 比 b1 劣等)
    #      因為 s2 劣等而 s, b 須交錯，所以刪除 s2
    while price[1-odd] < price[odd]:
        try:
            if price[2-2*odd] < price[2*odd]:
                price = np.delete(price, 0)
                date  = np.delete(date,  0)
                odd = 1-odd
            elif price[1+2*odd] < price[3-2*odd]:
                price = np.delete(price, [1, 2])
                date  = np.delete(date,  [1, 2])
            else:
                price = np.delete(price, [2, 3])
                date  = np.delete(date,  [2, 3])
        except:  # 進到這裡基本上是全虧損
            return np.array([]), np.array([])

    # 右端點的最佳化
    # 若   (len(price)+odd)%2 = 1，則 price[-1] 為 b，反之為 s
    # 假設 (len(price)+odd)%2 = 1  且 price[-4:] = s8, b8, s9, b9
    # 若最後一筆交易虧損 (s9-b9 < 0)，則必須刪除 s9, b9 其中之一
    # 接著找出刪除的可能性內最優等的方法
    # if   b8 < b9，則刪除 b9      (因為 b9 比 b8 劣等)
    # elif s9 < s8，則刪除 b8, s9  (因為 s9 比 s8 劣等)
    #      因為 b8 劣等而 s, b 須交錯，所以刪除 b8
    # else        ，則刪除 b8, s8  (因為 s8 比 s9 劣等)
    #      因為 b8 劣等而 s, b 須交錯，所以刪除 b8
    while price[-1-(len(price)+odd)%2] < price[-2+(len(price)+odd)%2]: # 尾
        try:
            if price[-1-(len(price)+odd)%2*2] < price[-3+(len(price)+odd)%2*2]:
                price = np.delete(price, -1)
                date  = np.delete(date,  -1)
            elif price[-4+(len(price)+odd)%2*2] < price[-2-(len(price)+odd)%2*2]:
                price = np.delete(price, [-2, -3])
                date  = np.delete(date,  [-2, -3])
            else:
                price = np.delete(price, [-3, -4])
                date  = np.delete(date,  [-3, -4])
        except:  # 進到這裡基本上是全虧損
            return np.array([]), np.array([])

    # 主體的最佳化
    # 若   sort = 1，則 price[index] 為 s，反之為 b
    # 假設 sort = 1  且 price[index:index+4] = s4, b4, s5, b5
    # 若中間的交易虧損 (s5-b4 < 0)，則必須刪除 b4, s5 其中之一
    # 接著找出刪除的可能性內最優等的方法
    # if   s4 < s5，則刪除 s4, b4  (因為 s4 比 s5 劣等)
    #      因為 s4 < s5 < b4，刪除 s5 不如刪除 s4
    # elif b4 < b5，則刪除 s5, b5  (因為 b5 比 b4 劣等)
    #      因為 s5 < b4 < b5，刪除 b4 不如刪除 b5
    # else        ，則刪除 b4, s5
    #      因為 s4, b5 都是更優等的選擇，所以選擇刪除 b4, s5
    index = 0
    while index < len(price)-3: # 中間
        sort = (index+odd)%2
        while index < len(price)-3 and price[index+2-sort] > price[index+1+sort]: 
            if price[index+2-2*sort] < price[index+2*sort]:
                price = np.delete(price, [index, index+1])
                date  = np.delete(date,  [index, index+1])
            elif price[index+3-2*sort] < price[index+1+2*sort]:
                price = np.delete(price, [index+2, index+3])
                date  = np.delete(date,  [index+2, index+3])
            else:
                price = np.delete(price, [index+1, index+2])
                date  = np.delete(date,  [index+1, index+2])        
        index += 1

    return date, price


if __name__ == "__main__":
    # 產生隨機交易資料
    # 怎麼產生的不重要，只要求買賣的漲跌勢相同
    time  = 100
    spread = 1.2
    sell = np.random.rand(1)  # 賣價
    buy  = np.array([sell[-1] * spread])  # 買價
    tick = np.random.randint(1, 5, 1)  # 交易時間
    for i in range(1, time):
        rand = (np.random.rand()-0.5) * (abs(np.random.rand())-0.5)
        next_price = rand * sell[-1] + sell[-1]
        sell = np.append(sell, next_price)
        buy  = np.append(buy,  sell[-1] * spread)
        tick = np.append(tick, tick[-1] + np.random.randint(1, 5))

    # 丟入買賣價，得到交易時機和交易價
    date, price = strategy(sell, buy, tick)  

    # 繪圖
    import matplotlib.pyplot as plt
    plt.plot(tick, sell, label="Selling price")
    plt.plot(tick, buy,  label="Purchase price")
    plt.legend()
    plt.plot(date, price, "s-")
    plt.show()