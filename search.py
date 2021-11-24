from model_cifar import *
c=Test()
ans = c.go('index2.jpeg')
import webbrowser

url1= "http://www.google.com/search?tbm=isch&q={}".format(ans)
webbrowser.open(url1,2)
url = "https://www.google.com.tr/search?q={}".format(ans)    
webbrowser.open(url,3)


