from math import exp as exp
from sklearn.metrics import roc_auc_score
from dataset import AdapTestDataset
from sklearn.metrics import accuracy_score
from collections import namedtuple
from scipy.optimize import minimize
from strategy.abstract_strategy import AbstractStrategy
from strategy.NCAT_nn.NCAT import NCATModel
from tqdm import tqdm

class NCATs(AbstractStrategy):

    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return 'NCAT'

    def adaptest_select(self, adaptest_data: AdapTestDataset,concept_map,config,test_length, path):
        selection = {}
        NCATdata = adaptest_data
        model = NCATModel(NCATdata,concept_map,config,test_length, path)
        threshold = config['THRESHOLD']
        for sid in tqdm(adaptest_data.data.keys(), "Training Policy: "):
 
            used_actions = []
            model.ncat_policy(sid,threshold,used_actions,type="training",epoch=100)
        NCATdata.reset()
        for sid in tqdm(adaptest_data.data.keys(), "Evaluating Policy: "):
            used_actions = []
            model.ncat_policy(sid,threshold,used_actions,type="testing",epoch=0)
            selection[sid] = used_actions
        NCATdata.reset()
        return selection
