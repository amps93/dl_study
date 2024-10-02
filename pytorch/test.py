from konlpy.tag import Komoran
from kiwipiepy import Kiwi
import time

k = Komoran()
kiwi = Kiwi(num_workers=4)
print(kiwi.num_workers)

start = time.time()
print(k.morphs('아빠가 방에 들어가신다'))
print(time.time()-start)

start = time.time()
print(kiwi.analyze('아빠가 방에 들어가신다'))
print(time.time()-start)

start = time.time()
print(kiwi.tokenize('아빠가 방에 들어가신다'))
print(time.time()-start)
