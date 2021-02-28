import random
import math
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['STZhongsong']   
mpl.rcParams['axes.unicode_minus'] = False          

class PSO(object):
    def __init__(self, num_city, data, num_group, iter_max, ui_obj):
        # 1. 初始化工作
        self.iter_max = iter_max       # 初始化迭代次数
        self.num_group = num_group  	 # 初始化种群数量
        self.num_city = num_city       # 初始化城市数量
        self.location = data           # 初始化城市的位置坐标
        self.ui_obj = ui_obj
        self.dis_mat = self.compute_dis_mat(num_city, self.location)  # 初始化城市两两之间的距离，得到一个距离矩阵        
        self.particals = self.random_init(self.num_group, num_city)   # 初始化种群中的所有粒子，给它们分配一个初始的最佳路径
        self.lenths = self.compute_paths(self.particals)        # 计算种群中每个粒子的路径长度
        # 2. 得到初始化群体的最优解
        init_l = min(self.lenths)                      # 计算种群中最短的路径长度
        init_index = self.lenths.index(init_l)         # 计算路径最短的粒子的索引
        init_path = self.particals[init_index]         # 根据索引找到最短的路径的城市序号
        # 3. 根据初始最优解，画出初始最佳路径图
        init_show = self.location[init_path]           # 根据城市序号找到对应城市的坐标
        plt.subplot(2, 2, 2)
        plt.title('初始最佳路径图')
        plt.xlabel('km')
        plt.ylabel('km')
        plt.plot(init_show[:, 0], init_show[:, 1],linestyle = '-',color = '#da16d5',marker = 's',markerfacecolor = '#18e225',markeredgecolor = 'black')
        # 4. 记录每个粒子的当前最佳路径和长度
        self.local_best = self.particals
        self.local_best_len = self.lenths
        # 5. 记录整个种群的当前最佳路径和长度
        self.global_best = init_path
        self.global_best_len = init_l
        # 6. 输出解
        self.best_l = self.global_best_len
        self.best_path = self.global_best
        # 7. 存储每次迭代的结果，画出收敛图
        self.iter_x = [0]
        self.iter_y = [init_l]

    # 随机初始化，返回一个二维数组，数组中每个元素代表每个粒子的初始最佳路径
    def random_init(self, num_total, num_city):
        tmp = [x for x in range(num_city)]
        result = []
        for i in range(num_total):
            random.shuffle(tmp)
            result.append(tmp.copy())
        return result

    # 计算各个城市两两之间的距离，返回一个距离矩阵
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算并返回单条路径的长度
    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算种群中每个粒子的路径长度，返回一个路径长度数组
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 评估当前的群体
    def eval_particals(self):
        min_lenth = min(self.lenths)                       # 找到此次迭代得到的全局最优解
        min_index = self.lenths.index(min_lenth)           # 找到全局最优解对应的索引
        cur_path = self.particals[min_index]               # 根据索引找到全局最优解对应的路径分布
        # 更新当前的全局最优
        if min_lenth < self.global_best_len:               # 如果比上次记录的全局最优解更优，则替换之
            self.global_best_len = min_lenth
            self.global_best = cur_path
        # 更新当前的个体最优
        for index, value in enumerate(self.lenths):        # 遍历每个粒子在此次迭代得到的路径长度
            if value < self.local_best_len[index]:         # 如果该路径长度比上次记录的局部最优解更优，则替换之
                self.local_best_len[index] = value
                self.local_best[index] = self.particals[index]

    # 粒子当前路径和自身的局部最优解路径进行交叉（需要处理重复情况）
    def cross(self, cur, best):
        one = cur.copy()
        l = [t for t in range(self.num_city)]                    
        t = np.random.choice(l,2)
        x, y = min(t), max(t)                                   # 随机选出两个城市序号
        cross_part = best[x:y]                                  # 在自身局部最优解路径中，取两个序号之间的序列
        tmp = []                                                # 直接交叉会导致城市序号重复，需要用temp存放出现在one中但没有出现在cross_part中的城市序号
        for t in one:                                           # 找出出现在one中但没有出现在cross_part中的城市序号
            if t in cross_part:
                continue
            tmp.append(t)
        # 两种交叉方法
        one = tmp + cross_part                                  # 两个序列的拼接有两种方式，即交叉有两种方式
        l1 = self.compute_pathlen(one, self.dis_mat)
        one2 = cross_part + tmp
        l2 = self.compute_pathlen(one2, self.dis_mat)
        if l1<l2:                                               # 哪种方式交叉后得到的路径较短，就返回其路径和长度
            return one, l1
        else:
            return one, l2


    # 粒子内部通过交换两个城市的位置，进行变异
    def mutate(self, one):
        one = one.copy()
        l = [t for t in range(self.num_city)]
        t = np.random.choice(l, 2)
        x, y = min(t), max(t)                               # 随机挑选出两个城市序号
        one[x], one[y] = one[y], one[x]                     # 在当前粒子的路径中，将这两个城市互换位置
        l2 = self.compute_pathlen(one,self.dis_mat)         # 计算互换位置（变异）后的新路径长度
        return one, l2                                      # 返回新路径及其长度

    # 迭代操作
    def pso(self):
        log_str = ''
        for cnt in range(1, self.iter_max + 1):
            # 更新粒子群：遍历每个粒子，每个粒子都要进行两次交叉和一次变异
            for i, one in enumerate(self.particals):
                tmp_l = self.lenths[i]
                # 1.第一次交叉：粒子与自身的当前局部最优解进行交叉
                new_one, new_l = self.cross(one, self.local_best[i])
                # 1.1 如果该粒子交叉后比全局更优，则替换之
                if new_l < self.best_l: 
                    self.best_l = tmp_l
                    self.best_path = one
                # 1.2 如果该粒子交叉后比上次更优或发生0.1概率的突变，则替换之    
                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l

                # 2.第二次交叉：粒子与当前全局最优解进行交叉
                new_one, new_l = self.cross(one, self.global_best)
                # 2.1 如果该粒子交叉后比全局更优，则替换之
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one
                # 2.2 如果该粒子交叉后比上次更优或发生0.1概率的突变，则替换之       
                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l
                # 3. 变异
                one, tmp_l = self.mutate(one)
                # 3.1 如果该粒子变异后比全局更优，则替换之
                if new_l < self.best_l:
                    self.best_l = tmp_l
                    self.best_path = one
                # 3.2 如果该粒子变异后比上次更优或发生0.1概率的突变，则替换之    	
                if new_l < tmp_l or np.random.rand()<0.1:
                    one = new_one
                    tmp_l = new_l

                # 更新该粒子
                self.particals[i] = one
                self.lenths[i] = tmp_l
            # 评估此次迭代后得到的新粒子群，计算出每个粒子的局部最优解和整个种群的全局最优解
            self.eval_particals()
            # 更新输出解
            if self.global_best_len < self.best_l:
                self.best_l = self.global_best_len
                self.best_path = self.global_best
            log_str += '第' + str(cnt) + '次迭代 | 最优解：' + str(self.best_l) + '\n'   
            self.ui_obj.textEdit.setText(log_str)    
            # print(f"第{cnt}次迭代 | 最优解：{self.best_l}")
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_l)

        return self.best_l, self.best_path

    def run(self):
    		# 1. 计算最佳路径及其长度
        best_length, best_path = self.pso()
        # 2. 绘制迭代结束后的最佳路径图（图3）
        plt.subplot(2, 2, 3)
        plt.title('迭代结束后的最佳路径图')
        plt.xlabel('km')
        plt.ylabel('km')
        plt.plot(self.location[best_path][:, 0],self.location[best_path][:, 1],linestyle = '-',color = '#da16d5',marker = 's',markerfacecolor = '#18e225',markeredgecolor = 'black')
        # 3. 画出迭代情况图（图4）
        plt.subplot(2, 2, 4)
        plt.title('迭代情况图')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度/最佳路径长度')
        plt.plot(self.iter_x, self.iter_y,linestyle = '-',color = '#da16d5')
        # return self.location[best_path], best_length
        return best_path, best_length