import streamlit as st 
import pandas as pd 
import plotly.express as px 
import talib 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
import matplotlib.pyplot as plt 
import graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.graph_objects as go
import json 
import datetime


# Utility function to parse the meta json file and get stock symbols. 
def parse_meta_json():

  with open('temporary-new.json', 'r') as f:
    s = f.read()

  meta_json = json.loads(s)
  stocks = meta_json["Stocks"]
  stock_to_symbol = dict() 
  for stock in stocks:
    stock_to_symbol[stock["Name"]] = {"Symbol": stock["Symbol"], "Data": stock["Data"], "Indicators": stock["Indicators"]}

  date = meta_json["Date"]
  today = datetime.date.today()
  year, month, day = map(int, date.split('-'))
  date_updated = datetime.date(year=year, month=month, day=day)

  if date_updated!=today:
    stocks = meta_json["Stocks"]
    for stock in stocks:
      stock["Data"] = 0 
      stock["Indicators"] = 0
    
    year, month, day = today.year, today.month, today.day

    date = f'{year}-{month}-{day}'
    new_meta_info = {}
    new_meta_info["Date"] = date 
    new_meta_info["Stocks"] = stocks
  
    with open('temporary-new.json', 'w') as f:
      f.write(json.dumps(new_meta_info, indent=4))

  return stock_to_symbol

# Function to update the meta json file after downloading and storing a csv from the API call. 
def update_meta_json(option, data=False, indicators=False):

  stocks = parse_meta_json()
  today = datetime.date.today()
  year, month, day = today.year, today.month, today.day
  date = f'{year}-{month}-{day}'
  
  meta_data = {}
  meta_data["Date"] = date 
  if data: 
    stocks[option]["Data"] = 1  
  if indicators:
    stocks[option]["Data"] = 1  
    stocks[option]["Indicators"] = 1  
  
  stocks_list = [] 
  for stock in stocks:
    st = dict()
    st["Name"] = stock 
    st.update(stocks[stock])
    stocks_list.append(st)

  meta_data["Stocks"] = stocks_list
  with open('temporary-new.json', 'w') as f:
    f.write(json.dumps(meta_data, indent=4))


# Used to set the CSS style for the HTML table.
st.markdown("""
<style>
table {
  font-family: arial, sans-serif;
  border-collapse: collapse;
  width: 100%;
}

td, th {

  text-align: left;
  padding: 8px;
}

tr:nth-child(even) {
  background-color: #dddddd;
}
</style>
""", unsafe_allow_html=True)



hello = st.sidebar.text('Home')
pwd = st.sidebar.text_input("Password", value="", type="password") 

if pwd == 'whatup': 
  st.sidebar.caption = 'Hello'
  
  stock_to_symbol = parse_meta_json()

  option = st.selectbox(
      'Pick Stock',
      tuple(stock_to_symbol.keys())
  )

  symbol = stock_to_symbol[option]["Symbol"]
  calculate_indicators = not(stock_to_symbol[option]["Indicators"])
  print('symbol: ', symbol)

  # If this value is 1 it means that we have downloaded the csv previously today and we can use that. 
  if stock_to_symbol[option]["Data"]:
    df = pd.read_csv(f'files/{symbol}.csv')
    df['timestamp'] = pd.to_datetime(df.timestamp)
    df = df.sort_values(by="timestamp")

  else:

    # API CALL to get data. 
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=BSE:{symbol}&datatype=csv&apikey=98O9B0WMAMRDDX0G'
    df = pd.read_csv(url)

    # Create an empty csv file on the fly. 
    with open(f'files/{symbol}.csv', 'w') as f:
      f.write(' ')
    
    # Sort data according to time.
    df['timestamp'] = pd.to_datetime(df.timestamp)
    df = df.sort_values(by="timestamp")
    df.to_csv(f'files/{symbol}.csv')
    print('Here, updating!')

    # Save Data. 
    update_meta_json(option, data=True)
    tech_df = df.copy() 
    
    
    # st.line_chart(data=(df['MACD'], df['MACD_Signal']))
  
  days = 10
  days = st.slider('# days to invest', min_value=5, max_value=60, value=35, step=5)

  if not calculate_indicators:
    tech_df_1 = df.copy() 
  
  else:
    # Bollinger Bands 
    upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    tech_df['bb_upper'] = upperband 
    tech_df['bb_middle'] = middleband 
    tech_df['bb_lowerband'] = lowerband
    
    # MACD
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    tech_df['macd'] = macd 
    tech_df['macd_signal'] = macdsignal
    tech_df['macdhist'] = macdhist
    tech_df['macd_useful'] = tech_df['macd']/tech_df['macd_signal']
    tech_df.head()

    # RSI 
    real = talib.RSI(df['close'], timeperiod=14)
    tech_df['rsi'] = real
    tech_df.head()

    # Stochastic Oscillator 
    slow_k, slow_d = talib.STOCH(df['high'], df['low'], df['close'])
    tech_df['stochastic_k'] = slow_k
    tech_df['stochastic_d'] = slow_d 
    tech_df['useful_stochastic'] = slow_k/slow_d
    tech_df.head()

    tech_df_1 = tech_df.dropna().copy()
    tech_df_1['returns'] = tech_df_1['close'].pct_change(days).shift(-1*days)
    tech_df_1.dropna(inplace=True)

    # Save tech indicators 
    tech_df_1.to_csv(f'files/{symbol}.csv')
    print('Here, tech updating!')
    update_meta_json(option, indicators=True)

  list_of_features = ['close', 'volume', 'bb_upper', 'bb_lowerband', 'macd_useful', 'macdhist', 'rsi', 'stochastic_k', 'stochastic_d', 'useful_stochastic']
  X = tech_df_1[list_of_features]
  y = np.where(tech_df_1.returns > 0, 1, 0)

  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=423)
  index = int(0.7*len(X))
  X_train, X_test, y_train, y_test = X[:index], X[index:], y[:index], y[index:]
  
  treeClassifier = DecisionTreeClassifier(min_samples_split=30, max_depth=15)
  treeClassifier.fit(X_train, y_train)
  y_pred = treeClassifier.predict(X_test)


  data = tree.export_graphviz(treeClassifier, filled=True, feature_names=list_of_features, class_names = np.array(['0', '1']))
  g = graphviz.Source(data)

  report = classification_report(y_test, y_pred)
  print(report)

  y_pred_temp = y_pred[-100:]
  X_closes = X_test['close'].iloc[-1*len(y_pred_temp):,]
  pred_df = pd.DataFrame()
  pred_df['Timestamp'] = tech_df_1['timestamp'].iloc[-1*len(y_pred_temp):, ]
  pred_df['Closes'] = X_closes
  pred_df['Call'] = np.where(y_pred_temp==0, 'Sell', 'Buy')
  
  closes = [] 
  calls = []

  for close, value in zip(list(pred_df['Closes'].values), list(pred_df['Call'].values)):
    closes.append(close)
    calls.append(value)

  profit = 0 
  units = []
  maximum = 0 
  overheads_per_sell = 0
  investments = (0,0) 
  returns = (0,0) 
  investments_remaining = (0,0)

  for i in range(len(X_closes)):
    if calls[i] == 'Sell':
      if len(units)!=0:
        k = 0
        while k<len(units) and units[k] <= i-days:
          profit += closes[i] - closes[units[k]]
          returns = returns[0]+closes[i],returns[1]+1
          investments_remaining = investments_remaining[0] - closes[units[k]], investments_remaining[1]-1
          k += 1 
        units = units[k:]
        profit -= overheads_per_sell
    else:
      investments = investments[0]+closes[i], investments[1]+1
      investments_remaining = investments_remaining[0]+closes[i], investments_remaining[1]+1
      
      units.append(i)

    maximum = max(maximum, sum(map(lambda x: closes[x], units)))

  returns = returns[0] + closes[-1]*investments_remaining[1], returns[1]+investments_remaining[1]
  print('Profit till now: ', profit)
  print('Investments remaining: ', investments_remaining)
  # plt.plot(pred_df['Timestamp'], pred_df['Closes'], 'k.-')
  # temp_df = pred_df[(pred_df.Call=="Buy")]
  # plt.plot(temp_df['Timestamp'], temp_df['Closes'], 'ro', color='red')
  # temp_df = pred_df[(pred_df.Call=="Sell")]
  # plt.plot(temp_df['Timestamp'], temp_df['Closes'], 'bo', color='blue')

  # st.write('{:30}{}'.format('Profit till now: ', profit))
  # st.write('{:30}{}'.format('Investments remaining: ', investments_remaining))
  # st.write('Investment: ', investments)
  # st.write('Returns: ', returns)
  # st.write( f'Profit: {round((returns[0]-investments[0]) / investments[0]* 100,2)}% in {len(y_pred_temp)} days')
  # st.write('Maximum Investment: ', maximum)

  st.markdown(f'''
    <table>
      <tr>
        <th>Attribute</th>
        <th>Value</th>
      </tr>
      <tr>
        <td>Profit Till Now</td>
        <td>Rs. {round(profit,2)}</td>
      </tr>
      
      <tr>
        <td>Investments remaining to square off</td>
        <td>Rs. {round(investments_remaining[0], 2)} | {investments_remaining[1]} units </td>
      </tr>

      <tr>
        <td>Total Investments</td>
        <td>Rs. {round(investments[0],2)} | {investments[1]} units </td>
      </tr>

      <tr>
        <td>Total Returns</td>
        <td>Rs. {round(returns[0],2)} | {returns[1]} units </td>
      </tr>

      <tr>
        <td>Profit percentage</td>
        <td>{round((returns[0]-investments[0]) / investments[0]* 100,2)}% in {len(y_pred_temp)} days</td>
      </tr> 
      <tr>
        <td>Maximum Investment</td>
        <td>Rs. {round(maximum,2)}</td>
      </tr>
      
        
    </table>
  ''', unsafe_allow_html=True)

  fig = px.line(pred_df, x='Timestamp', y=['Closes'])
  fig.update_traces(line_color='#456987')
  color_dm = {''}
  fig2 = px.scatter(pred_df, x="Timestamp", y="Closes", color="Call")
  fig3 = go.Figure(data=fig.data+fig2.data)
  fig3.update_layout(
      xaxis=dict(
          rangeselector=dict(
              buttons=list([
                  dict(count=7,
                      label="7d",
                      step="day",
                      stepmode="backward"),
                  dict(count=1,
                      label="1m",
                      step="month",
                      stepmode="backward"),
                  dict(count=6,
                      label="6m",
                      step="month",
                      stepmode="backward"),
                  dict(count=1,
                      label="YTD",
                      step="year",
                      stepmode="todate"),
                  dict(count=1,
                      label="1y",
                      step="year",
                      stepmode="backward"),
                  dict(step="all")
              ])
          ),
          rangeslider=dict(
              visible=True
          ),
          type="date"
      )
  )

  st.plotly_chart(fig3)
  st.write(report)