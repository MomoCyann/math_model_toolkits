import pandas as pd
a = {
   'x' : 1,
   'y' : 2,
   'z' : 3
}

b = {
   'w' : 9,
   'x' : 5,
   'y' : 2
}

# 交集
print(a.keys() & b.keys())
# 并集
c = a.keys() | b.keys()
print(c)
c.remove('x')
print(c)
# 差集
a.keys() - b.keys()
b.keys() - a.keys()