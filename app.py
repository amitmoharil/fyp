import streamlit as st 
import pandas as pd 
import talib 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
import numpy as np 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from util import *

stock_to_symbol = parse_meta_json()

option = st.selectbox(
    'Pick Stock',
    tuple(sorted(stock_to_symbol.keys()))
)

symbol = stock_to_symbol[option]["Symbol"]
calculate_indicators = not(stock_to_symbol[option]["Indicators"])


# If this value is 1 it means that we have downloaded the csv previously today and we can use that. 
if stock_to_symbol[option]["Data"]:
  df = pd.read_csv(f'files/{symbol}.csv')
  df['timestamp'] = pd.to_datetime(df.timestamp)
  df = df.sort_values(by="timestamp")

else:

  # API CALL to get data. 
  url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=BSE:{symbol}&datatype=csv&apikey=AKWG9C3GNQ9NBD32'
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
  
  ltp = df.at[df.index[-1], 'close']
  print(ltp)

  # ------------------------ #
  # Bollinger 
  upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
  tech_df['bb_upper'] = upperband 
  tech_df['bb_middle'] = middleband 
  tech_df['bb_lowerband'] = lowerband
  tech_df['bb_percentage'] = (df['close'] - tech_df['bb_lowerband']) /(tech_df['bb_upper']-tech_df['bb_lowerband']+1) * 100

  # ----------------------- #
  # MACD
  macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
  tech_df['macd'] = macd 
  tech_df['macd_signal'] = macdsignal
  tech_df['macdhist'] = macdhist
  tech_df['macd_percentage'] = tech_df['macdhist']/tech_df['macd_signal'] * 100
  tech_df.head()
  tech_df.tail()

  # ---------------------- #
  # RSI 
  real = talib.RSI(df['close'], timeperiod=14)
  tech_df['rsi'] = real
  tech_df.head()

  # ---------------------- #
  # Stochastic Oscillator 
  slow_k, slow_d = talib.STOCH(df['high'], df['low'], df['close'])
  tech_df['stochastic_k'] = slow_k
  tech_df['stochastic_d'] = slow_d 
  tech_df['stochastic_percentage'] = (slow_k-slow_d) * 100
  tech_df.head()

  # ---------------------- #
  # EMA

  tech_df['ma_44'] = talib.SMA(df['close'], timeperiod=44)
  tech_df['ema_200'] = talib.EMA(df['close'], timeperiod=200)

  tech_df['useful_ma_44_percentage'] = (df['close']-tech_df['ma_44'])/df['close'] * 100
  tech_df['useful_ema_200_percentage'] = (df['close']-tech_df['ema_200'])/df['close'] * 100

  # ---------------------- #
  # Volume MA 
  tech_df['volume'] = df['volume']
  tech_df['ma_volume'] = talib.SMA(df['volume'], timeperiod=20)
  tech_df['ma_volume_percentage'] = (tech_df['ma_volume']-tech_df['volume']) / tech_df['volume'] * 100
  
  # ---------------------- #
  # ATR
  tech_df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

  tech_df_1 = tech_df.dropna().copy()
  tech_df_1['returns'] = tech_df_1['close'].pct_change(days).shift(-1*days)
  tech_df_2 = tech_df_1.iloc[-1*days:]
  tech_df_1.dropna(inplace=True)

# Creating a list of features to pass to the decision tree. 
list_of_features = ['macd_percentage', 'bb_percentage', 'rsi', 'stochastic_percentage', 'useful_ema_200_percentage', 'ATR', 'ma_volume_percentage']
tech_df_1.dropna(inplace=True)
index = int(0.85*len(tech_df_1))
X = tech_df_1[list_of_features].iloc[:index]
y = np.where(tech_df_1.returns > 0, 1, 0)[:index]
tech_df_1['Calls'] = np.where(tech_df_1.returns > 0, 1, 0)
print(tech_df_1.Calls.value_counts())

# Splitting training and testing datasets.
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=45)

# Creating and Fitting Decision Tree Classifier
treeClassifier = DecisionTreeClassifier(min_samples_split=20, max_depth=15)
treeClassifier.fit(X_train, y_train)
y_pred = treeClassifier.predict(X_test)

# Printing the accuracy, precision, recall. 
report = classification_report(y_test, y_pred, output_dict=True)


# Profit Calculation given closes and calls
# Add resistance and support here. 
def profit_calculator(pred_df):
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


index = -200
tech_df_1['Calls'] = np.where(tech_df_1['Calls']==0, 'Sell', 'Buy')
df = tech_df_1[:]
y_pred_temp = treeClassifier.predict(tech_df_1[list_of_features][index:])
X_closes = tech_df_1['close'].iloc[index:]
pred_df = pd.DataFrame()
pred_df['timestamp'] = tech_df_1['timestamp'].iloc[-1*len(y_pred_temp):, ]
pred_df['close'] = X_closes
pred_df['Calls'] = np.where(y_pred_temp==0, 'Sell', 'Buy')
profit_calculator(pred_df)


with st.container():
  st.write('\n\n')
  col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
  col1.button(label=f'{symbol:20}', key='1')
  col2.button(label=f'Rs. {ltp}', key='2')
  col3.button(label='Positive', key='3')
  col4.button(label='Buy', key='4')

  
plot_calls(df)
plot_calls(pred_df)

report["Sell"] = report["0"]
report["Buy"] = report["1"]
del report["0"], report["1"]
st.table(pd.DataFrame(report))

local_css("css\style.css")