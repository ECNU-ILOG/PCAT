#——————————————————————————————————————————————————————————————

#该文件解释说明：此代码用于进化算法过程中试卷对象的定义和变异函数等内容的定义

#——————————————————————————————————————————————————————————————


import numpy as np
import torch
import random
from dataset.adaptest_dataset import AdapTestDataset 
from model.IRT import IRTModel
from model.NCD import NCDModel
from sklearn.metrics import mean_squared_error
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
import copy
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import asyncio
from collections import defaultdict
from dataset.train_dataset import TrainDataset

def cal_COV_EXP(question_overlap, concept_map, metadata, test_length, selected_probs, test_stu):
    stu_coverages = {}
    stu_overlaps = {}
    for stu in test_stu:
        concepts = set()   #包含的知识点集合
        overlap = 0     #曝光率
        for prob in selected_probs[stu]:
            overlap += (question_overlap[prob] - test_length/metadata['prob_num']) ** 2
            concepts.update(concept_map[prob])
        coverage = len(concepts)/metadata['know_num']    #知识点覆盖率
        overlap = overlap/(test_length/metadata['prob_num'])  #曝光率
        stu_coverages[stu] = coverage
        stu_overlaps[stu] = overlap
        cov = sum(stu_coverages.values())/len(stu_coverages)
        exp = sum(stu_overlaps.values())/len(stu_overlaps)
    return cov, exp

def adjust_to_int(x, question_type, num_selected):#用于调整题目类型不重复
    int_array = x.astype(int)
    unique_integers = np.unique(int_array)
    #print('selected:',num_selected)
    if len(unique_integers) != num_selected:
        flag = True
    else:
        flag = False
    #if flag == True:
        #print('出现重复的题目类型:',x)
    max_type = max(question_type.keys())+1
    for i in range(num_selected):
        x[i] = int(x[i])
        if flag == True or x[i] == -1:
            copy_x = x[:num_selected].copy()
            copy_x = np.delete(copy_x, i)
            while x[i] in copy_x:
                x[i] = np.random.randint(0, max_type)
    #if flag == True:
        #print('删除重复题目类型后的结果为:',x)


class CustomCrossover(Crossover):   #交叉操作

    def __init__(self, num_selected, question_type, a = 2, b = 2):
        super().__init__(a, b)  # 父代数量和子代数量
        self.num_selected = num_selected
        self.question_type = question_type

    def _do(self, problem, X, **kwargs):
        #print('进行了一次交叉')
        # X 的形状为 (n_parents, n_matings, n_var)
        n_parents, n_matings, n_var = X.shape
        Y = np.empty_like(X)

        for i in range(n_matings):
            #针对题目类型进行随机交叉
            #print('i:',i)
            parent1, parent2 = X[:, i, 0:self.num_selected]
            adjust_to_int(parent1, self.question_type, self.num_selected)
            adjust_to_int(parent2, self.question_type, self.num_selected)

            #print('parent1:',parent1)
            #print('parent2:',parent2)

            start, end = sorted(np.random.choice(self.num_selected, 2, replace=False))  
            
            offspring1 = np.full(self.num_selected, -1)  # 用 -1 表示空
            offspring2 = np.full(self.num_selected, -1)

            # 复制交叉区间的部分到子代
            offspring1[start:end] = parent1[start:end]
            offspring2[start:end] = parent2[start:end]

            #print('offspring1:',offspring1)
            #print('offspring2:',offspring2)

            # 填充子代的非交叉区间
            offspring1 = self._adjust(offspring1, parent2, end)
            offspring2 = self._adjust(offspring2, parent1, end)
            
            Y[0, i, 0:self.num_selected] = offspring1
            Y[1, i, 0:self.num_selected] = offspring2
            
            #print('结果2:')
            #print('offspring1:',offspring1)
            #print('offspring2:',offspring2)

            #print('结果3:')
            #print('Y1:',Y[0, i, 0:self.num_selected])
            #print('Y2:',Y[1, i, 0:self.num_selected])

            #------------------------------------------------------
            #针对题目难度进行交叉
            parent1, parent2 = X[:, i, self.num_selected:]
            start, end = sorted(np.random.choice(self.num_selected, 2, replace=False))   
            offspring1 = parent1.copy() 
            offspring2 = parent2.copy()

            # 复制交叉区间的部分到子代
            offspring1[start:end] = parent2[start:end]
            offspring2[start:end] = parent1[start:end]

            Y[0, i, self.num_selected:] = offspring1
            Y[1, i, self.num_selected:] = offspring2

        return Y
    
    def _adjust(self, offspring, parent, end): 
        current_pos = end
        n_var = self.num_selected
        for i in range(len(offspring)):
            if offspring[i] == -1:
                while int(parent[current_pos % n_var]) in offspring:
                    current_pos += 1
                #print('位置',i,':',offspring[i],'->',parent[current_pos%n_var])
                offspring[i] = int(parent[current_pos%n_var])

        return offspring


class CustomMutation(Mutation): #变异
    def __init__(self, num_selected, question_type, maxmin_difficulty, prob_mutation=0.5):
        super().__init__()
        self.num_selected = num_selected
        self.prob_mutation = prob_mutation
        self.question_type = question_type
        self.maxmin_difficulty = maxmin_difficulty  #保存题目难度的最大值和最小值

    def _do(self, problem, X, **kwargs):
        #print('进行了一次变异')
        for i in range(X.shape[0]):
            adjust_to_int(X[i][:self.num_selected], self.question_type, self.num_selected)
            for j in range(self.num_selected):
                if np.random.rand() < self.prob_mutation:   #对于题目类型的变异
                    max_type = max(self.question_type.keys())+1
                    x = (X[i][0:self.num_selected]).copy()
                    x = np.delete(x, j)
                    while True:
                        type = np.random.randint(0, max_type)
                        if float(type) not in x:
                            break
                    X[i][j] = type

                if np.random.rand() < self.prob_mutation:   #对于具体题目的变异
                    X[i][j+self.num_selected] =  X[i][j+self.num_selected] + np.random.normal(loc=0.0, scale=0.5)
                    #X[i][j+self.num_selected] = np.random.uniform(self.maxmin_difficulty[1],self.maxmin_difficulty[0])
        return X

    def _adjust_to_constraints(self, x):    #此处暂时用不到调整函数，用于调整变异操作后的结果是否满足要求
        return x


class Paper(ElementwiseProblem):  #构造试卷子代
    def __init__(self, paper_config):
        # 定义问题的维度和目标数量
        #paper_config参数,max_selected,n_dim, k_constraint, e_constraint
        super().__init__(n_var=paper_config['num_selected']*2, 
                        n_obj=4, 
                        xl=np.array([0] * paper_config['num_selected'] + [np.min(paper_config['question_difficulty'])] * paper_config['num_selected'], dtype=float),
                        xu=np.array([max(paper_config['question_type'].keys())] * paper_config['num_selected'] + [np.max(paper_config['question_difficulty'])] * paper_config['num_selected'], dtype=float),
                        type_var= float)
        
        self.metadata = paper_config['metadata']   #用于保存学生、题目、知识点等数量
        self.num_selected = paper_config['num_selected']    #最多可以选择的题目数量
        self.concept_map = paper_config['concept_map']  #保存题目-知识点字典
        self.question_overlap = paper_config['question_overlap'] #保存各题目的曝光率
        self.model_true = paper_config['model_true']   #保存哟关于评估的CDM
        self.train_stu = paper_config['train_stu'] #保存学生训练集
        self.ckpt_path = paper_config['ckpt_path']   #用于保存初始化CDM的路径
        self.question_difficulty = paper_config['question_difficulty'] #用于保存每道题的难度表征
        self.device = paper_config['device']   #用于保存设备
        self.question_type = paper_config['question_type'] #用于保存题目类型和各自包含的题目
        self.cdm = paper_config['cdm']
        self.seed = paper_config['seed']
        self.config = paper_config['config']
        self.train_triplets = paper_config['train_triplets']

    def calculate_ACC(self, questions):
        question_list = []
        for prob in questions:
            prob_difficulty = self.question_difficulty[prob]
            prob_difficulty = np.mean(prob_difficulty[prob_difficulty != 0])
            question_list.append((prob, prob_difficulty))
        
        sorted_questions = sorted(question_list, key=lambda x: x[1])
        sorted_question_indices = [item[0] for item in sorted_questions]
        paper_length = len(questions)//3
        paper1 = sorted_question_indices[0:paper_length] #简单
        paper2 = sorted_question_indices[paper_length:2*paper_length] #中等
        paper3 = sorted_question_indices[2*paper_length:]   #难

        new_train_data = AdapTestDataset(self.train_triplets, self.concept_map,
                                        self.metadata['stu_num'],
                                        self.metadata['prob_num'], 
                                        self.metadata['know_num'], 
                                        self.seed)
        new_train_data.reset2()

        stu_cov = {}
        stu_exp = {}
        for stu in self.train_stu:
            concepts = set()   #包含的知识点集合
            overlap = 0     #曝光率
            p = 0
            for prob in paper2:
                new_train_data.apply_selection(stu, prob)
                overlap += (self.question_overlap[prob] - (self.num_selected - 10)/self.metadata['prob_num']) ** 2
                concepts.update(self.concept_map[prob])
                if new_train_data.data[stu][prob] == 1:
                    p += 1
            p = p/paper_length
            if p >= 0.5:
                for prob in paper3:
                    new_train_data.apply_selection(stu, prob)
                    overlap += (self.question_overlap[prob] - (self.num_selected - 10)/self.metadata['prob_num']) ** 2
                    concepts.update(self.concept_map[prob])
            else:
                for prob in paper1:
                    new_train_data.apply_selection(stu, prob)
                    overlap += (self.question_overlap[prob] - (self.num_selected - 10)/self.metadata['prob_num']) ** 2
                    concepts.update(self.concept_map[prob])
            stu_cov[stu] = len(concepts)/self.metadata['know_num']    #知识点覆盖率
            stu_exp[stu] = overlap/((self.num_selected - 10)/self.metadata['prob_num'])  #曝光率
            new_train_data.meta[stu] = [que for que in range(0, self.metadata['prob_num']) if que not in new_train_data._tested[stu]]

        cov = sum(stu_cov.values())/len(stu_cov)
        exp = sum(stu_exp.values())/len(stu_exp)
        

        #构造初始model
        if self.cdm == 'irt':
            model = IRTModel(**(self.config))
        elif self.cdm == 'ncd':
            model = NCDModel(**(self.config))
        model.init_model(new_train_data)
        model.adaptest_load(self.ckpt_path)
        #model.config['num_epochs'] = 6
        #model.config['batch_size'] = 64
        model.update(new_train_data, out=False)
        
        outinformation = model.evaluate(new_train_data) 
        #print('ACC:',outinformation['acc'], ', AUC:',outinformation['auc'],)      
        return outinformation['acc'], outinformation['auc'], cov, exp


    def _evaluate(self, x, out, *args, **kwargs):
        questions = []  #先构造空白试卷
        #print('原来的paper矩阵:',x)
        #print(type(x))
        adjust_to_int(x[:self.num_selected], self.question_type, self.num_selected)
        #print('Paper评估:',x)
        for i in range(self.num_selected):#将子代中的题目矩阵转化为具体题目列表
            n = -1
            for j in self.question_type[int(x[i])]:
                #print('当前j:',j)
                if n == -1:
                    n = j
                else:
                    que1 = self.question_difficulty[n]
                    que2 = self.question_difficulty[j]
                    #print('前类型:',type(np.mean(que1[que1 != 0])))
                    #print('后类型:',type(x[i+self.num_selected]))
                    if abs(np.mean(que1[que1 != 0]) - x[i+self.num_selected]) > abs(np.mean(que2[que2 != 0]) - x[i+self.num_selected]):
                        n = j
            questions.append(n)
        acc, auc, cov, exp = self.calculate_ACC(questions)
        #print('mse:',mse)

        # 目标：最大化覆盖率（负数化以适应最小化框架），最小化重叠率
        out["F"] = [-cov, exp, -acc, -auc]
        