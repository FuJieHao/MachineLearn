import redis
import urllib.request
import re

rconn = redis.Redis("172.17.0.2", '6379')

#url  http://www.17k.com/book/2801661.html
'''
url - i - '1'
'''
for i in range(0, 4801661):
    isdo = rconn.hget("url",str(i))
    if(isdo != None):
        continue
    rconn.hset("url", str(i), "1")
    try:
        data =  urllib.request.urlopen("http://www.17k.com/book/" + str(i) + ".html").read().decode("utf-8", "ignore")
    except Exception as err:
        print(str(i) + err)
        continue

    #<a class="red" href="/book/2801661.html">血色佳人</a>
    pat = '<a class="red" .*?>(.*?)</a>'
    rst = re.compile(pat, re.S).findall(data)
    if(len(rst) == 0):
        continue
    name = rst[0]
    rconn.hset('rst',str(i),str(name))