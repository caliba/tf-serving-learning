#-*- encoding = utf-8 -*-
#@Time :2021/12/22 19:12
#@Author : Agonsle
#@File :restful_timeshow.py
#@Software :PyCharm

import  xlrd
import matplotlib.pyplot as plt
import  numpy as np
x = xlrd.open_workbook(r'data_r&r.xlsx')

a = [i+1 for i in range(49)]
table = x.sheets()[0]
col = table.col_values(2,1,50)
col = [round(x*1000,2) for x in col]

plt.plot(a,col)
print(col)
plt.title('gRPC execution time')
plt.xlabel("Number of requests /n")
plt.ylabel("Request time*1000 /ms")
plt.show()