from bs4 import BeautifulSoup
import requests
  
url = "https://www.pathtoschools.com/"
  
html_content = requests.get(url).text
home_soup = BeautifulSoup(html_content, "html.parser")
states = home_soup.findAll("a", {"class":"region1"})

for state in states[:2]:

    state_url = url + state['href'] 
    html_content = requests.get(state_url).text
    state_soup = BeautifulSoup(html_content, "html.parser")
    regions = state_soup.findAll("a", {"class": "region1"})
    print('State: ', state.text)
    
    for region in regions:
        
        region_url = region["href"]
        html_content = requests.get(region_url).text
        region_soup = BeautifulSoup(html_content, "html.parser")
        pages = len(region_soup.find('div', {'class':'pagination'}).findAll('a'))
        
        print('Region: ', region.text)
        print('#Pages: ', pages) 

        for i in range(1, pages+1):
            page_url = region_url + f"/{i}"
            html_content = requests.get(region_url).text
            region_soup = BeautifulSoup(html_content, "html.parser")
            schools = region_soup.find('table', {"class": 'school-list'}).findAll('tr')[2:]
            for school in schools[:1]:
                datas = school.findAll('td')    
                try:         
                    print('School Name: ', datas[1].find('h3').text)
                    print('School Phone: ', datas[2].find('span').text.strip())
                    add1 = datas[1].find('h3').next_sibling
                    add2 = add1.next_sibling
                    print('School Address: ',  add1.text.strip() + ", " + add2.next_sibling.text.strip())
                except:
                    print('Failed')
                    pass 