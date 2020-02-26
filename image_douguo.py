import os
from urllib import parse
import requests # install
from bs4 import BeautifulSoup as bs # install

#https://www.douguo.com/search/recipe/%E7%B3%96%E9%86%8B%E6%8E%92%E9%AA%A8/0/0
douguo_api='https://www.douguo.com/search/recipe/'
keyword='鱼香茄子'
headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.109 Safari/537.36'}
path='douguo'


def download(lis,path):
	number=1
	c_path=path+'\\'+keyword
	if not os.path.exists(path):
		os.mkdir(path)
	if not os.path.exists(c_path):
		os.mkdir(c_path)
	for url in lis:
		r=requests.get(url,headers=headers,stream=True)
		with open(c_path+'\\'+str(number)+'.jpg','wb') as f:
			for chunk in r.iter_content(chunk_size=32):
				f.write(chunk)
		print(str(number)+'.'+url+' downloaded!')
		number=number+1
	print('downloading successfully!')


def main():
	global keyword
	k=parse.quote(keyword)
	page=11
	lis=[]
	for p in range(page):
		url=douguo_api+k+'/0/'+str(20*p)
		r=requests.get(url,headers=headers)
		soup=bs(r.text,'html.parser')
		cook_list=soup.find('ul',class_='cook-list')
		for li in cook_list.find_all('li'):
			style=li.find('a',class_='cook-img').get('style')
			img_url=style.split('(')[1].split(')')[0] #字符串分割
			#print(img_url)
			lis.append(img_url)
	download(lis, path)



if __name__ == '__main__':
	main()
