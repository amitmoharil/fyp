'''
1. Clustering youtube videos - playlist 
2. Stock - values seen before 
3. 
'''

import streamlit as st 
import pandas as pd 
import plotly.express as px
import talib 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
import graphviz
from sklearn import tree
from sklearn.metrics import classification_report
import plotly.graph_objects as go
import json 
import datetime
from sklearn.model_selection import train_test_split


# Utility function to parse the meta json file and get stock symbols. 
def parse_meta_json():

  with open('temporary-new.json', 'r') as f:
    s = f.read()

  meta_json = json.loads(s)
  stocks = meta_json["Stocks"]
  stock_to_symbol = dict() 
  for stock in stocks:
    stock_to_symbol[stock["Name"]] = {"Symbol": stock["Symbol"], "Data": stock["Data"], "Indicators": stock["Indicators"], "Industry": stock["Industry"]}

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

# Printing Buy/Sell calls to visualise where exactly we are buying or selling.
def plot_calls(df):
  dcm = {'Buy':'#618a4d', 'Sell':'#0000ff'}
  fig = px.line(df, x='timestamp', y=['close'])
  fig.update_traces(line_color='#456987')
  fig2 = px.scatter(df, x="timestamp", y="close", color="Calls", color_discrete_map=dcm)
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

# Used to set the CSS style for the HTML table.
st.markdown(
"""
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


# SideBar details can go here. 
hello = st.sidebar.text('Home')
pwd = st.sidebar.text_input("Password", value="", type="password") 


stock_to_symbol = parse_meta_json()

option = st.selectbox(
    'Pick Stock',
    tuple(sorted(stock_to_symbol.keys()))
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
  
  
  # st.line_chart(data=(df['MACD'], df['MACD_Signal']))

days = st.slider('# Days to Invest', min_value=5, max_value=60, value=35, step=5)
print(days)

ltp = df.at[df.index[-1], 'close']
amount_to_invest = st.number_input('Amount to Invest', min_value=0.0, max_value=1000000.0, value=ltp, step=0.01)

# For now this must never be true. 
if not calculate_indicators:
  tech_df_1 = df.copy() 
else:

  # Bollinger Bands 
  tech_df = df.copy() 
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

  # RSI 
  real = talib.RSI(df['close'], timeperiod=14)
  tech_df['rsi'] = real

  # Stochastic Oscillator 
  slow_k, slow_d = talib.STOCH(df['high'], df['low'], df['close'])
  tech_df['stochastic_k'] = slow_k
  tech_df['stochastic_d'] = slow_d 
  tech_df['stochastic_useful'] = slow_k/slow_d

  # MA
  ma = talib.MA(tech_df['close'], timeperiod=44)
  tech_df['ma_44'] = ma 
  tech_df['useful_ma_44'] = tech_df['close'] - ma

  # OBV 
  tech_df['obv'] = talib.OBV(df['close'], df['volume'])

  # ATR 
  tech_df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

  tech_df_1 = tech_df.dropna().copy()
  tech_df_1['returns'] = tech_df_1['close'].pct_change(days).shift(-1*days)
  tech_df_2 = tech_df_1.iloc[-1*days:]
  tech_df_1.dropna(inplace=True)

# Creating a list of features to pass to the decision tree. 
list_of_features = ['ATR', 'macd_useful', 'bb_upper', 'bb_lowerband', 'macdhist', 'rsi', 'stochastic_useful', 'useful_ma_44', 'obv']
tech_df_1.dropna(inplace=True)
index = int(0.85*len(tech_df_1))
X = tech_df_1[list_of_features].iloc[:index]
y = np.where(tech_df_1.returns > 0, 1, 0)[:index]
tech_df_1['Calls'] = np.where(tech_df_1.returns > 0, 1, 0)
print(tech_df_1.Calls.value_counts())

# Splitting training and testing datasets.
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=45)

# Creating and Fitting Decision Tree Classifier
treeClassifier = DecisionTreeClassifier(min_samples_split=30, max_depth=15)
treeClassifier.fit(X_train, y_train)
y_pred = treeClassifier.predict(X_test)


# Printing the accuracy, precision, recall. 
report = classification_report(y_test, y_pred, output_dict=True)

# Profit calculation 
index = -500
y_pred_temp = treeClassifier.predict(tech_df_1[list_of_features][index:])
X_closes = tech_df_1['close'].iloc[index:]
pred_df = pd.DataFrame()
pred_df['timestamp'] = tech_df_1['timestamp'].iloc[-1*len(y_pred_temp):, ]
pred_df['close'] = X_closes
pred_df['Calls'] = np.where(y_pred_temp==0, 'Sell', 'Buy')

closes = [] 
calls = []

for close, value in zip(list(pred_df['close'].values), list(pred_df['Calls'].values)):
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


tech_df_1['Calls'] = np.where(tech_df_1['Calls']==0, 'Sell', 'Buy')
df = tech_df_1[:]
plot_calls(df)
plot_calls(pred_df)

report["Sell"] = report["0"]
report["Buy"] = report["1"]
del report["0"], report["1"]
st.table(pd.DataFrame(report))

'''# Trying to compare analyse separation of calls. 
col1 = st.selectbox(
    'Pick Feature 1',
    tuple(list(tech_df_1.columns)[2:-1])
)

col2 = st.selectbox(
    'Pick Feature 2',
    tuple(list(tech_df_1.columns)[2:-1])
)

col3 = st.selectbox(
    'Pick Feature 3',
    tuple(list(tech_df_1.columns)[2:-1])
)

fig = px.scatter_3d(data_frame=tech_df_1, x=col1, y=col2, z=col3, color='Calls')
st.plotly_chart(fig)
'''
# Plotting the Decision Tree 
# Keeping a fixed depth - otherwise too big. 
data = tree.export_graphviz(treeClassifier, filled=True, feature_names=list_of_features, class_names = {0:'Sell', 1:'Buy'}, out_file=None, max_depth=3)
st.graphviz_chart(data, use_container_width=True)

'''
# Report for the 20% of the data we did neither training or testing on. 
report = classification_report(tech_df_1['Calls'][index:], pred_df['Calls'], output_dict=True)
st.table(pd.DataFrame(report))
'''