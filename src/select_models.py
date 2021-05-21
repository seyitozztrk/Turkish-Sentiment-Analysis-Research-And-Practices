from basic_abc import basic_operation
from utils import utils 

class models(basic_operation):

    def __init__(self):

        super().__init__()
        
    def getXY(self):
        
        x,y = self.getNormalizedData()
        
        return  x,y
    
    
    def run(self):
        
        x,y = self.getNormalizedData()
        xtrain, _, _, _ = self.splitKFoldStratified(x,y)

    def svmModel(self):
#         self.embedding_process()
        self.train_test_split_dataset('svm')

    def NaiveBayesModel(self):
        
        self.train_test_split_dataset('bayes')
  
    def DecisionTreeModel(self):
       
        self.train_test_split_dataset('decision')
 