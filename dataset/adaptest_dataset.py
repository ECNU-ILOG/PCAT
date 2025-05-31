from collections import defaultdict, deque
import torch
import random

try: 
 
    from .dataset import Dataset
    from .train_dataset import TrainDataset
except (ImportError, SystemError): 
 
    from dataset import Dataset
    from train_dataset import TrainDataset


class AdapTestDataset(Dataset): 

    def __init__(self, data, concept_map,   
                 num_students, num_questions, num_concepts, seed): 
        """
        Args:
            data: list, [(sid, qid, score)]
            concept_map: dict, concept map {qid: cid}
            num_students: int, total student number
            num_questions: int, total question number
            num_concepts: int, total concept number
        """
        super().__init__(data, concept_map,
                         num_students, num_questions, num_concepts)

 
        self.candidate = None 
        self.meta = None 
        self._tested = None 
        self._untested = None 
        self.seed = seed
        self.reset() 

    def apply_selection(self, student_idx, question_idx): 
        """ 
        Add one untested question to the tested set
        Args:
            student_idx: int
            question_idx: int
        """
        #if question_idx not in self._untested[student_idx]:
        #    print('缺失的学生:',student_idx,'题目:',question_idx)
        #    print(self._untested[student_idx])
        #assert question_idx in self._untested[student_idx], 'Selected question not allowed'
        if question_idx in self._untested[student_idx]:
            self._untested[student_idx].remove(question_idx) 
        self._tested[student_idx].append(question_idx) 

    def reset(self): 
        """ 
        Set tested set empty
        """
        self.candidate = dict() 
        self.meta = dict() 
        for sid in self.data: 
            random.seed(self.seed) 
            self.candidate[sid] = random.sample(self.data[sid].keys(), int(len(self.data[sid]) * 0.8)) 
            self.meta[sid] = [log for log in self.data[sid].keys()]
            #self.meta[sid] = [log for log in self.data[sid].keys() if log not in self.candidate[sid]] 
        self._tested = defaultdict(deque) 
        self._untested = defaultdict(set) 
        for sid in self.data: 
            self._untested[sid] = set(self.candidate[sid]) 

    def reset2(self): #自定义一个函数重新构造候选集
        """ 
        Set tested set empty
        """
        self.candidate = dict() 
        self.meta = dict() 
        for sid in self.data: 
            self.candidate[sid] = self.data[sid].keys() 
        self._tested = defaultdict(deque) 
        self._untested = defaultdict(set) 
        for sid in self.data: 
            self._untested[sid] = set(self.candidate[sid]) 

    @property
    def tested(self): 
        return self._tested

    @property
    def untested(self): 
        return self._untested

    def get_tested_dataset(self, last=False, ssid=None): 
        """
        Get tested data for training
        Args: 
            last: bool, True - the last question, False - all the tested questions
        Returns:
            TrainDataset
        """
        if ssid==None: 
            triplets = [] 
            for sid, qids in self._tested.items(): 
                if last: 
                    qid = qids[-1] 
                
                    triplets.append((sid, qid, self.data[sid][qid])) 
                else: 
                    for qid in qids: 
                        #print('sid:',sid)
                        #print('qid:',qid)
                        #print('correct:',self.data[sid][qid])
                        triplets.append((sid, qid, self.data[sid][qid])) 
            return TrainDataset(triplets, self.concept_map, 
                                self.num_students, self.num_questions, self.num_concepts) 
        else:
            triplets = [] 
            for sid, qids in self._tested.items(): 
                if ssid == sid: 
                    if last: 
                        qid = qids[-1] 
                        triplets.append((sid, qid, self.data[sid][qid])) 
                    else: 
                        for qid in qids: 
                            triplets.append((sid, qid, self.data[sid][qid])) 
            return TrainDataset(triplets, self.concept_map,
                                self.num_students, self.num_questions, self.num_concepts) 
        
    def get_meta_dataset(self): 
        triplets = {} 
        for sid, qids in self.meta.items(): 
            triplets[sid] = {} 
            for qid in qids: 
                triplets[sid][qid] = self.data[sid][qid] 
        return triplets 