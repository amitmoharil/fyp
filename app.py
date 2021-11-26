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

hello = st.sidebar.text('Home')
pwd = st.sidebar.text_input("Password", value="", type="password") 

option = st.selectbox(
    'Pick Stock',
    ('Relicance Industries Limited', 'Adani Ports', 'Tata Consultancy Services', 'Larsen and Toubro')
)
stock_to_symbol = {'Relicance Industries Limited':'RELIANCE', 'Adani Ports': 'ADANIPORTS', 'Tata Consultancy Services': 'TCS', 'Larsen and Toubro': 'LT'}

symbol = stock_to_symbol[option]
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=BSE:{symbol}&datatype=csv&apikey=98O9B0WMAMRDDX0G'
df = pd.read_csv(url)
df['timestamp'] = pd.to_datetime(df.timestamp)
df = df.sort_values(by="timestamp")
tech_df = df.copy() 
print(df.head())
print(df.tail())
# st.line_chart(data=(df['MACD'], df['MACD_Signal']))
days = 10
days = st.slider('# days to invest', min_value=5, max_value=60, value=10, step=5)


# Bollinger Bands 
upperband, middleband, lowerband = talib.BBANDS(df['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
tech_df['bb_upper'] = upperband 
tech_df['bb_middle'] = middleband 
tech_df['bb_lowerband'] = lowerband
print(tech_df.head())

# MACD
macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
tech_df['macd'] = macd 
tech_df['macd_signal'] = macdsignal
tech_df['macdhist'] = macdhist
tech_df.head()

# RSI 
real = talib.RSI(df['close'], timeperiod=14)
tech_df['rsi'] = real
tech_df.head()

tech_df_1 = tech_df.dropna().copy()
tech_df_1['returns'] = tech_df_1['close'].pct_change(days).shift(-1*days)
print(tech_df_1.head())
print(tech_df_1.tail())
list_of_features = ['close', 'open', 'high', 'low', 'volume', 'macd', 'bb_upper', 'bb_lowerband', 'macd_signal', 'macdhist', 'rsi']
tech_df_1.dropna(inplace=True)
X = tech_df_1[list_of_features]

print(X.head())
y = np.where(tech_df_1.returns > 0, 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=423)

treeClassifier = DecisionTreeClassifier(max_depth=5)
treeClassifier.fit(X_train, y_train)
y_pred = treeClassifier.predict(X_test)

data = tree.export_graphviz(treeClassifier, filled=True, feature_names=list_of_features, class_names = np.array(['0', '1']))
g = graphviz.Source(data)

print(X_train)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

treeClassifier = DecisionTreeClassifier(max_depth=5)
treeClassifier.fit(X_train, y_train)
y_pred = treeClassifier.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

y_pred_temp = y_pred[-200:]

X_closes = tech_df['close'].iloc[-1*len(y_pred_temp):,]
pred_df = pd.DataFrame()
pred_df['Timestamp'] = tech_df['timestamp'].iloc[-1*len(y_pred_temp):, ]
pred_df['Closes'] = X_closes
pred_df['Call'] = np.where(y_pred_temp==0, 'Sell', 'Buy')
pred_df.tail(200)

closes = [] 
calls = []

for close, value in zip(list(pred_df['Closes'].values), list(pred_df['Call'].values)):
  closes.append(close)
  calls.append(value)

profit = 0 
units = []
maximum = 0 
overheads_per_sell = 20
investments = (0,0) 
returns = (0,0) 
investments_remaining = (0,0)

for i in range(len(X_closes)):
  if calls[i] == 'Sell':
    if len(units)!=0:
      k = 0
      while units[k] <= i-days:
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

print(investments_remaining)
returns = returns[0] + closes[-1]*investments_remaining[1], returns[1]+investments_remaining[1]
print('Profit till now: ', profit)
print('Investments remaining: ', investments_remaining)
# plt.plot(pred_df['Timestamp'], pred_df['Closes'], 'k.-')
# temp_df = pred_df[(pred_df.Call=="Buy")]
# plt.plot(temp_df['Timestamp'], temp_df['Closes'], 'ro', color='red')
# temp_df = pred_df[(pred_df.Call=="Sell")]
# plt.plot(temp_df['Timestamp'], temp_df['Closes'], 'bo', color='blue')

st.write('{:30}{}'.format('Profit till now: ', profit))
st.write('{:30}{}'.format('Investments remaining: ', investments_remaining))
st.write('Investment: ', investments)
st.write('Returns: ', returns)
st.write( f'Profit: {round((returns[0]-investments[0]) / investments[0]* 100,2)}% in {len(y_pred_temp)} days')
st.write('Maximum Investment: ', maximum)



fig = px.line(pred_df, x='Timestamp', y=['Closes'])
fig.update_traces(line_color='#456987')
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