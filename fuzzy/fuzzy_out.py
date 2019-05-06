

class Reout(object):

    def shuchu(self,dengjia_matrix):
        qqq =input("请输入聚类的阈值（大于0小于1）")
        u =float(qqq)
        W = dengjia_matrix
        num =W.shape[1]
        for i in range(num):
            for j in range(num):
                if W[i, j] < u:
                    W[i, j] = 0
                else:
                    W[i, j] = 1

        print("结果矩阵为：")
        print(W)

        # # 根据结果矩阵得到聚类结果
        # group = num
        # # print(group)
        # list1 = []
        # list2 = []
        # for j in range(num):
        #     if (W[0, j] == 1):
        #         group = group - 1
        #         list2.append(j + 1)
        #     else:
        #         list3 = []
        #         list3.append(j + 1)
        #         list1.append(list3)
        # list1.append(list2)
        # print("聚类出" + str(group+1) + "组,结果如下所示：")
        # print(list1)