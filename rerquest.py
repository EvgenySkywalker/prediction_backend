import requests

# 35.228.184.173 http://www.fayloobmennik.net/files/go/422604494.html?check=cdd69a1ed2ec6192fa0faadeb3514e89&file=5863010
a = requests.post('http://127.0.0.1:8000/data/annotate', data='csv/sdasdasdas/dsds.csv')

print(a.text)
