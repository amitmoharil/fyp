import json 
import pandas as pd
import time 

with open('Equity.csv') as f:
    lines = f.readlines()[1:]

stocks = []
filename = 'temporary-new-new-new.json'
d = {"Date":"2021-04-07", "Stocks":[]}
for i, line in enumerate(lines[:]):
    data = line.split(',')
    if len(data[1])>0:    
        print(data[1])
        try:
            current = {"Name":data[1], "Symbol":data[0], "Data":0, "Indicators":0,"Industry":data[-2]}
                
            symbol1 = current["Symbol"]
            time.sleep(60)
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=BSE:{symbol1}&datatype=csv&apikey=98O9B0WMAMRDDX0G'
            df = pd.read_csv(url)
            df['timestamp'] = pd.to_datetime(df.timestamp)
            d["Stocks"].append(current)
        except:
            try:
                symbol2 = data[2].strip()
                url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=BSE:{symbol2}&datatype=csv&apikey=98O9B0WMAMRDDX0G'
                df = pd.read_csv(url)
                df['timestamp'] = pd.to_datetime(df.timestamp)
                current["Symbol"] = symbol2
                d["Stocks"].append(current)
            except:
                stocks.append((symbol1, symbol2))
                print('HEREEE:', symbol1, symbol2)
                
                pass
        
        time.sleep(60)
        print(f'{i} done')
        
with open(filename, 'w') as f:
    f.write(json.dumps(d, indent=4))

print("\n".join(stocks))