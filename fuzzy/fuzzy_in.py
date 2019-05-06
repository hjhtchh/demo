import csv
import numpy as np
class Rein(object):

    def shuru(self):
        #输入一个路径读取特征矩阵，并输出
        # s =input("请输入一个特征矩阵所在的路径:")
        # print(s)
        my_matrix =np.loadtxt(open('D:/consumption2.csv',encoding='utf-8'),delimiter="," ,skiprows=0)
        #print(my_matrix)
        return my_matrix


# if __name__ == '__main__':
#     k = Rein()
#     k.shuru()