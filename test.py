import datetime
import json

def parse_meta_json():
  
  with open('temporary.json', 'r') as f:
    s = f.read()

  meta_json = json.loads(s)
  stock_to_symbol = meta_json["Stocks"]
  date = meta_json["Date"]
  today = datetime.date.today()
  year, month, day = map(int, date.split('-'))
  date_updated = datetime.date(year=year, month=month, day=day)

  if date_updated!=today:
    stock = meta_json["Stocks"]
    for key in stock:
      stock[key][1] = 0
    
    year, month, day = today.year, today.month, today.day

    date = f'{year}-{month}-{day}'
    new_meta_info = {}
    new_meta_info["Date"] = date 
    new_meta_info["Stocks"] = stock
  
    with open('temporary.json', 'w') as f:
      f.write(json.dumps(new_meta_info))

    stock_to_symbol = stock

  return stock_to_symbol

def main():
    parse_meta_json()

if __name__ == '__main__':
    main()