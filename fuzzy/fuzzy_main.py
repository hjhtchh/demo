from fuzzy import  fuzzy_in
from fuzzy import  fuzzy_method
from fuzzy import  fuzzy_out
class FuzzyMain(object):

    #初始化所需的参数
    def __init__(self,):
       self.rein = fuzzy_in.Rein()
       self.selectmethod = fuzzy_method.Methods()
       self.reout = fuzzy_out.Reout()

    def start(self):
        #获得特征矩阵
        in_matrix=self.rein.shuru()
        print("输入的特征矩阵为：")
        print(in_matrix)
        #获得需要聚类的个体数量
        num = in_matrix.shape[1]
        print("聚类的个体数量为：")
        print(num)
        #获得模糊相似矩阵
        print("---------------------")
        print("选择建立相似关系的方法")
        print("1.绝对值减数法")
        print("2.数量积法")
        a =int(input("请输入你要选择的方法前的编号"))
        print("----------------------------")
        fuzzy_matrix=[]
        if a == 1:
            fuzzy_matrix = self.selectmethod.absmethod(in_matrix)
        #运用自乘法求传递闭包得到模糊等价矩阵
        dengjia_matrix = self.selectmethod.Cdbb(fuzzy_matrix)
        print("模糊等价矩阵为：")
        print(dengjia_matrix)
        #根据输入不同的阈值获得不同的聚类结果
        self.reout.shuchu(dengjia_matrix)

if __name__ == '__main__':
    begin =FuzzyMain()
    begin.start()