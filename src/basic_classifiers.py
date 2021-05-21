
class basic_operation:
  
    def splitKFoldStratified( X, y):
        from sklearn.model_selection import StratifiedKFold
        
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        
        Xtrain_array = []
        Xtest_array = []
        Ytrain_array = []
        Ytest_array = []
        
        k_fold = 0 
        
        for train_ix, test_ix in kfold.split(X, y):
            k_fold+=1
            # select rows
            xtrain, xtest = X[train_ix], X[test_ix]
            ytrain, ytest = y[train_ix], y[test_ix]
            
            Xtrain_array.append(xtrain)
            Xtest_array.append(xtest)
            Ytrain_array.append(ytrain)
            Ytest_array.append(ytest)
            
            # summarize train and test composition
            train_0, train_1 = len(ytrain[ytrain==0]), len(ytrain[ytrain==1])
            test_0, test_1 = len(ytest[ytest==0]), len(ytest[ytest==1])
            print('>Train: 0=%d, 1=%d, Test: 0=%d, 1=%d' % (train_0, train_1, test_0, test_1))
        
        
        return [Xtrain_array, Xtest_array, Ytrain_array, Ytest_array]
    


    def generateTfIdfVector(_x):
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=10000, analyzer='word')

        x_train = vectorizer.fit_transform(_x)
        return x_train
    
    
    
        
    def runSVM(X_train, X_test, y_train, y_test):
        import numpy as np 
        from sklearn import svm
        linear_svm = svm.LinearSVC()
        
        linear_svm.fit(X_train, y_train)
        y_pred = linear_svm.predict(X_test)
        
        visualizeConfusionMatrix(y_test, y_pred)
        showScores(y_test, y_pred)
        
    def runBayes(X_train, X_test, y_train, y_test):
        
        from sklearn.naive_bayes import MultinomialNB
        sentiment_model = MultinomialNB().fit(X_train, y_train)
        y_pred = sentiment_model.predict(X_test)
        
        visualizeConfusionMatrix(y_test, y_pred)
        showScores(y_test, y_pred)

    def runDecisionTree(X_train, X_test, y_train, y_test):
        from sklearn import tree
        decision_tree = tree.DecisionTreeClassifier()
        decision_tree.fit(X_train, y_train)

        y_pred = decision_tree.predict(X_test)

        visualizeConfusionMatrix(y_test, y_pred)
        showScores(y_test, y_pred)


        
        
        
        
        
        
        
        
        
        
        
        
        
    



  
