import os
import urllib.request

os.makedirs("data", exist_ok=True)

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "data/tinyshakespeare.txt")

print("Downloaded to data/tinyshakespeare.txt")