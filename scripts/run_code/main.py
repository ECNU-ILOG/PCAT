import sys
import os
sys.path.append('../..')
import argparse
import torch
import numpy as np
import pandas as pd
import random
#import datetime
import json
from scripts.dense_paper import *
#from tqdm import tqdm
from collections import defaultdict
from strategy.NCAT_strategy import NCATs
from strategy.random_strategy import RandomStrategy
from strategy.MFI_strategy import MFIStrategy
from strategy.KLI_strategy import KLIStrategy
from strategy.MAAT_strategy import MAATStrategy
from strategy.BECAT_strategy import BECATstrategy
from strategy.BOBCAT_strategy import BOBCAT
from model.IRT import IRTModel
from model.NCD import NCDModel
from dataset.adaptest_dataset import AdapTestDataset 
from dataset.train_dataset import TrainDataset
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
#import matplotlib.pyplot as plt
from data.data_params_dict import data_params
import scripts.bobcat_train as Bobcatcode
from model.utils import StraightThrough
import time
import warnings
warnings.filterwarnings("ignore")

class MyCallback(Callback):
    def __init__(self,output_file, interval=10):
        super().__init__()
        self.output_file = output_file
        self.interval = interval
        self.iteration_count = 0 

    def notify(self, algorithm, **kwargs):
        self.iteration_count += 1
        F = algorithm.pop.get("F")
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        if self.iteration_count % self.interval == 0:
            j = 0
            X = algorithm.pop.get("X")
            with open(self.output_file, 'a') as file:
                file.write('The' + str(self.iteration_count) + 'output:\n')
                for i in nds:
                    file.write('son:'+str(j)+':'+','.join(map(str, X[i])) + '  ,objectives:' + ','.join(map(str, F[i])) + '\n')  
                    j += 1
                file.write('\n')

        max_acc = 0
        for i in nds:
            max_acc = max(max_acc, -1*F[i][2])
        print('In the current iteration step, the maximum value of ACC is:',max_acc)

def cal_COV_EXP(question_overlap, concept_map, metadata, test_length, selected_probs, test_stu):
    stu_coverages = {}
    stu_overlaps = {}
    for stu in test_stu:
        concepts = set()  
        overlap = 0   
        for prob in selected_probs[stu]:
            overlap += (question_overlap[prob] - test_length/metadata['prob_num']) ** 2
            concepts.update(concept_map[prob])
        coverage = len(concepts)/metadata['know_num']    
        overlap = overlap/(test_length/metadata['prob_num'])  
        stu_coverages[stu] = coverage
        stu_overlaps[stu] = overlap
        cov = sum(stu_coverages.values())/len(stu_coverages)
        exp = sum(stu_overlaps.values())/len(stu_overlaps)
    return cov, exp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='Dense_Assistment17', type=str, help='benchmark')
    parser.add_argument('--cdm', default='ncd', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--test_length', type=int, default=20)
    parser = parser.parse_args()

    seed_list = [20]
    strategy_list = ["mynew"]
    print('start')
    script_name = os.path.basename(__file__)
    print("script_name:", script_name)
    print('dataset:',parser.dataset)
    print('Model:',parser.cdm)
    dataset = parser.dataset
    # modify config here
    config = {
        'learning_rate': 0.05, 
        'batch_size': 512, 
        'num_epochs': 10,  
        'num_dim': 1, 
        'device': parser.device,
        'cdm': parser.cdm,
        # for NeuralCD
        'prednet_len1': 128,  
        'prednet_len2': 64,
        # for BOBCAT
        'betas': (0.9, 0.999),
        'policy_path': '../../model/ckpt/{}_bocat_policy.pt'.format(parser.dataset),
        # for NCAT
        'THRESHOLD' :300,
        'start':0,
        'end':3000,
        'seed':20
    }

    base_model_config = {
        'learning_rate': 0.01,
        'batch_size': 1024,
        'num_epochs': 10, 
        'num_dim': 1,
        'device': parser.device,
        'cdm': 'irt',
        'betas': (0.9, 0.999),
        'seed':20
    }

    # modify checkpoint path here
    ckpt_path1 = f'../../model/ckpt/model_{dataset}_all_{parser.cdm}.pt'
    ckpt_path2 = f'../../model/ckpt/model_{dataset}_train_{parser.cdm}.pt'
    #bobcat_policy_path =config['policy_path']

    # read datasets
    response_logs = pd.read_csv(f'../../data/{dataset}/TotalData.csv', header=None)
    response_logs = response_logs.drop_duplicates()
    q = pd.read_csv(f'../../data/{dataset}/q.csv', header=None)
    with open('../../data/{}/question_map.json'.format(dataset), 'r') as f:
        question_map = json.load(f)
    metadata = {
    'stu_num': data_params[dataset]['stu_num'],
    'prob_num': data_params[dataset]['prob_num'],
    'know_num': data_params[dataset]['know_num'],
    }
    
    if dataset.replace("Dense_","") == "MOOCRadar" or dataset.replace("Dense_","") == "Assistement17":
        original_response_logs = pd.read_csv(f'../../data/{dataset.replace("Dense_","")}/TotalData.csv', header=None)
    else:
        original_response_logs = pd.read_csv(f'../../data/{dataset.replace("Dense_","")}/TotalData.csv')
    question_overlap = defaultdict(int)
    for log in original_response_logs.values:
        if str(int(log[1])) in question_map.keys():
            question_overlap[question_map[str(int(log[1]))]] += 1

    for prob in question_overlap.keys():
        question_overlap[prob] = question_overlap[prob]/data_params[dataset.replace("Dense_","")]['stu_num']

    concept_map = {}
    cnt = 0
    for question in q.values:
        concept_map[cnt] = np.where(np.array(question))[0].tolist()
        cnt += 1

    with open(f'{parser.dataset}_output.txt', 'w', encoding='utf-8') as file:
            file.write("The dataset for the current output file is:" +parser.dataset + '\n')  # 每个对象写入一行

    for seed in seed_list:
        with open(f'{parser.dataset}_output.txt', 'a', encoding='utf-8') as file:
            file.write("seed:" + str(seed) + '\n') 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        total_stu = list(range(0, metadata['stu_num']))
        train_stu = random.sample(total_stu, int(len(total_stu) * 0.8)) 

        test_stu = [stu for stu in total_stu if stu not in train_stu]  
        print("test_stu:",test_stu)
        train_triplets = [] 
        test_triplets = [] 
        for log in response_logs.values:
            if int(log[0]) in train_stu:
                train_triplets.append((int(log[0]), int(log[1]), int(log[2])))
            else:
                test_triplets.append((int(log[0]), int(log[1]), int(log[2])))

        train_data = TrainDataset(train_triplets, concept_map,
                                            metadata['stu_num'],
                                            metadata['prob_num'], 
                                            metadata['know_num'])
    
        if parser.cdm == 'irt':
            model_train = IRTModel(**config)
        elif parser.cdm == 'ncd':
            model_train = NCDModel(**config)
        model_train.init_model(train_data)

        model_train.train(train_data,out=True)
        model_train.adaptest_save(ckpt_path2)

        base_model = IRTModel(**base_model_config)
        base_model_path = f'../../model/ckpt/model_{parser.dataset}_train_irt.pt'
        base_model.init_model(train_data)

        base_model.train(train_data, out=True)
        base_model.adaptest_save(base_model_path)
        
        strategyname = "mynew"
        config['policy'] = "mynew"
        train_start_time = time.time()
        question_difficulty = model_train.model.k_difficulty.weight.data.cpu().numpy().copy()
        question_difficulty = question_difficulty * q
        question_difficulty = question_difficulty.to_numpy()

        question_type = defaultdict(list)
        know_hash = {}
        cnt = 0
        for que, knows in concept_map.items():
            sorted_knows = tuple(sorted(knows))
            if sorted_knows not in know_hash.keys():
                know_hash[sorted_knows] = cnt
                cnt += 1
            question_type[know_hash[sorted_knows]].append(que)

        paper_config = {
                'metadata' : metadata,  
                'num_selected' : int(parser.test_length*3/2),  
                'concept_map' : concept_map,    
                'question_overlap' : question_overlap,  
                'model_true' : None, 
                'train_stu' : train_stu,   
                'ckpt_path' : ckpt_path2,
                'question_difficulty' : question_difficulty,    
                'device' : parser.device,
                'question_type': question_type, 
                'cdm' : parser.cdm, 
                'seed' : seed,
                'config' : config,
            'train_triplets':train_triplets
            }

        algorithm = NSGA2(pop_size = 100,
                        n_offsprings = 50, 
                        sampling = FloatRandomSampling(), 
                        crossover=CustomCrossover(num_selected = paper_config['num_selected'], question_type = question_type, a = 2, b = 2),
                        mutation=CustomMutation(num_selected = paper_config['num_selected'], question_type = question_type, maxmin_difficulty=[question_difficulty.max(),question_difficulty.min()], prob_mutation=0.5),
                        eliminate_duplicates=True)

        res = minimize(Paper(paper_config),
                        algorithm,
                        ('n_gen',80),
                        seed=11,
                        verbose=True,
            )

        # Deep copy the result and release memory
        difficulty_paper = copy.deepcopy(res.pop)
        del res

        def get_paper(x, paper_config):
            questions = []  

            for i in range(paper_config['num_selected']):
                n = -1
                for j in paper_config['question_type'][int(x[i])]:
                    if n == -1:
                        n = j
                    else:
                        que1 = paper_config['question_difficulty'][n]
                        que2 = paper_config['question_difficulty'][j]
                        if abs((que1[que1 != 0]).mean() - x[i+paper_config['num_selected']]) > abs((que2[que2 != 0]).mean() - x[i+paper_config['num_selected']]):
                            n = j
                questions.append(n)
            return questions

        # Get objective function values and decision variables
        F = difficulty_paper.get("F")
        X = difficulty_paper.get("X")
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        cnt = -1
        max_auc_acc = 0
        best_paper = []
        with open(f'{parser.dataset}_output.txt', 'a', encoding='utf-8') as file:
            file.write(strategyname+ "method" + "step" + str(parser.test_length*3/2)+ " output:\n" + "all papers:\n")  
            for i in range(0, len(nds)):
                paper_candidate = get_paper(X[nds[i]].tolist(),paper_config)
                file.write("paper:" + str(paper_candidate) + "  objectives:" + str(F[nds[i]].tolist()) + "\n")
                if cnt == -1 or max_auc_acc < (F[nds[i]].tolist())[3]*-1 + (F[nds[i]].tolist())[2]*-1:
                    cnt = nds[i]
                    best_paper = paper_candidate
                    max_auc_acc = (F[nds[i]].tolist())[3]*-1 + (F[nds[i]].tolist())[2]*-1
        train_end_time = time.time()

        test_start_time = time.time()
        question_list = []
        for prob in best_paper:
            prob_difficulty = question_difficulty[prob]
            prob_difficulty = np.mean(prob_difficulty[prob_difficulty != 0])
            question_list.append((prob, prob_difficulty))

        sorted_questions = sorted(question_list, key=lambda x: x[1])
        sorted_question_indices = [item[0] for item in sorted_questions]
        paper_length = len(best_paper)//3
        paper1 = sorted_question_indices[0:paper_length] 
        paper2 = sorted_question_indices[paper_length:2*paper_length] 
        paper3 = sorted_question_indices[2*paper_length:]   


        
        new_test_data = AdapTestDataset(test_triplets, concept_map,
                                metadata['stu_num'],
                                metadata['prob_num'], 
                                metadata['know_num'],
                                seed)   
        new_test_data.reset2()

        if parser.cdm == 'irt':
            model = IRTModel(**config)
        elif parser.cdm == 'ncd':
            model = NCDModel(**config)
        model.init_model(new_test_data)
        model.adaptest_load(ckpt_path2)
                
        stu_cov = {}
        stu_exp = {}
        for stu in test_stu:
            concepts = set() 
            overlap = 0     
            p = 0
            for prob in paper2:
                new_test_data.apply_selection(stu, prob)
                overlap += (question_overlap[prob] - parser.test_length/metadata['prob_num']) ** 2
                concepts.update(concept_map[prob])
                if new_test_data.data[stu][prob] == 1:
                    p += 1
            p = p/paper_length
            if p >= 0.5:
                for prob in paper3:
                    new_test_data.apply_selection(stu, prob)
                    overlap += (question_overlap[prob] - parser.test_length/metadata['prob_num']) ** 2
                    concepts.update(concept_map[prob])
            else:
                for prob in paper1:
                    new_test_data.apply_selection(stu, prob)
                    overlap += (question_overlap[prob] - parser.test_length/metadata['prob_num']) ** 2
                    concepts.update(concept_map[prob])

            stu_cov[stu] = len(concepts)/metadata['know_num']    
            stu_exp[stu] = overlap/(parser.test_length/metadata['prob_num'])  
            new_test_data.meta[stu] = [que for que in range(0, metadata['prob_num']) if que not in new_test_data._tested[stu]]

        cov = sum(stu_cov.values())/len(stu_cov)
        exp = sum(stu_exp.values())/len(stu_exp)

            
        model.update(new_test_data, out=False)

        outinformation = model.evaluate(new_test_data)
        test_end_time = time.time()
        auc = outinformation['auc']
        acc = outinformation['acc']

        with open(f'{parser.dataset}_output.txt', 'a', encoding='utf-8') as file:
            file.write("The selected final paper:" + str(best_paper) +"\nThe evaluation results on the test set: AUC:"+ str(auc) + "ACC:"+ str(acc) + "COV:"+ str(cov) + "EXP:"+ str(exp) + ", online:"+  str(train_end_time - train_start_time) + "s, offonline time:"+ str(test_end_time - test_start_time) + "s" + '\n')  # 每个对象写入一行

                





