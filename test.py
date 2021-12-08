import json 
import datetime 
import random

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
  
  print(stocks[option])
  stocks_list = [] 
  for stock in stocks:
    st = dict()
    st["Name"] = stock 
    st.update(stocks[stock])
    stocks_list.append(st)

  meta_data["Stocks"] = stocks_list
  print(meta_data)
  with open('temporary-new.json', 'w') as f:
    f.write(json.dumps(meta_data, indent=4))


if __name__ == '__main__':
  d = parse_meta_json()
  choice = random.choice(list(d.keys()))
  print(choice)

  update_meta_json(choice, indicators=True)