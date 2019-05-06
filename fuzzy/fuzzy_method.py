import  numpy as np
class Methods(object):

    #绝对值减数法
    def absmethod(self,matrix):
        num = matrix.shape[1]
        fuzzy = np.ones((num,num))*1
        c= 0.0001
        for i in range(num):
            for j in range(num):
                q = 1 - abs(matrix[:, i] - matrix[:, j]).sum() * c
                fuzzy[i, j] = q
        print("模糊相似矩阵为：")
        print(fuzzy)
        return fuzzy

    #自乘法求传递闭包
    def Cdbb(self,fuzzy_matrix):
        flag = 0
        num=fuzzy_matrix.shape[1]
        R = fuzzy_matrix
        B = np.zeros((num, num))
        while flag == 0:
            for i in range(num):
                for j in range(num):
                    for k in range(num):
                        B[i, j] = max(min(R[i, k], R[k, j]), B[i, j])

            if (B == R).all():
                flag = 1
            else:
                R = B
        return R