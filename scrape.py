# Writing a python script for getting the shakespeare's play
import requests
from bs4 import BeautifulSoup
def get_file(url,title):
	response_= requests.get(url)
	soup_= BeautifulSoup(response_.content,'html.parser')
	p_tags= soup_.find_all('p', style=True)
	with open(f"{title}.txt",'a') as file:
		for ptag in p_tags:
			file.write(ptag.text+'\n')
	file.close()
def get_urls(url_main):
	response= requests.get(url_main)
	soup= BeautifulSoup(response.content,'html.parser')
	div_title= soup.find_all('div', class_= 'summary-title')
	title_, url_ = [], []
	for title in div_title:
		title_.append(title.text)
		a= title.find('a')
		try:
			if 'href' in a.attrs:
				url= a.get('href')
				url_.append(url)
		except:
			print("Exception Occurred")
	return url_, title_

def main():
	url_main= 'https://www.thecompleteworksofshakespeare.com'
	urls, titles= get_urls(url_main)
	# urls1, titles1= urls[42:55], titles[42:55]
	# for url,title in zip(urls1,titles1):
	# 	url_= url_main+url
	# 	get_file(url_, title[1:])
	print(titles)
	print(len(titles))

	
if __name__=="__main__":
	main()
    
