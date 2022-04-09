import pandas as pd 
import requests
for _ in range(10):
    symbol1 = 'RELIANCE'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=BSE:{symbol1}&datatype=csv&apikey=98O9B0WMAMRDDX0G'
    r = requests.get(url)
    print(r.text)
    df = pd.read_csv(url)
    print(df)