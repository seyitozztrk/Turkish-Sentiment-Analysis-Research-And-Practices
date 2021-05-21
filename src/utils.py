class utils:

    def getAllData():
        import pandas as pd 
        print('init')
        #fetch data from csv files
        pos = pd.read_csv('/content/drive/MyDrive/lstmWorks/pos.txt', delimiter="\t", header=None)
        neg = pd.read_csv('/content/drive/MyDrive/lstmWorks/neg_reviews.txt', delimiter="\t", header=None)
        #positive samples
        pos['target'] = 1
        pos.columns=['text', 'target']
        #negative samples
        neg['target'] = 0
        neg.columns=['text', 'target']

        df = pd.concat([neg, pos], ignore_index=True)
        df.reset_index()
        return df
        
        

    def getXyData():
        print('hello')
        import nltk
        nltk.download('stopwords')
        import re 
        import numpy as np 
        import pickle
        
        from TurkishStemmer import TurkishStemmer
        from snowballstemmer import stemmer

        df = utils.getAllData()

        WPT = nltk.WordPunctTokenizer()
        stop_word_list = nltk.corpus.stopwords.words('turkish')
        
        stemmer = TurkishStemmer()

        try:
            file = open('normalization_X_data.pickle', 'rb')
            X = pickle.load(file)
        except:

            X = np.asarray([], dtype=object)
            print("->>" , type(X))
            for i in range(0, len(df)):
                yorum = re.sub("[^AaBbCcÇçDdEeFfGgĞğHhİiIıJjKkLlMmNnOoÖöPpRrSsŞşTtUuÜüVvYyZz']", ' ', df['text'][i])
                yorum = re.sub("[']", '', yorum) #drop things that without letters
                yorum = yorum.lower()
                yorum = yorum.strip()
                yorum = yorum.split()

                yorum = [stemmer.stem(word) for word in yorum if word not in stop_word_list]
                yorum = ' '.join(yorum)

                X = np.append(X,[yorum])
                

            pickle.dump(X, open("normalization_X_data.pickle", "wb"))

        y = df['target'].to_numpy()
    
        return X,y



    def makeEmbeddingMatrix(embeddingDimension,typeOfEmbed, pathOfEmbed, word_index, max_features=25000 ):
    
        from gensim.models import KeyedVectors
        import os 
        import numpy as np
        global word_vectors
        total_words = len(word_index) + 1

        if typeOfEmbed == 'word2vec':

          word_vectors = KeyedVectors.load_word2vec_format(pathOfEmbed, binary=True)
      
        elif typeOfEmbed == 'glove':

          word_vectors = {}
          f = open(pathOfEmbed)
          for line in f:
              values = line.split()
              word = values[0]
              coefs = np.asarray(values[1:], dtype='float32')
              word_vectors[word] = coefs
          f.close()

      
        embedding_matrix = np.zeros((max_features, embeddingDimension))
        embedding_vector = None
        
        for word, index in word_index.items():
            if index >= max_features: break
            try:
                embedding_vector = word_vectors[word]
            except:
                pass
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

        print("Embeddings Matrix shape : ",embedding_matrix.shape)
        return embedding_matrix



    # Now, let’s display the training and validation loss and accuracy

    # Plotting results
    def vizualize_loss_acc(history):
        
        import matplotlib.pyplot as plt
        acc = history.history['acc']
        val_acc = history.history['val_acc']


        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')

        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()

        # *******************************************************************
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()    




    def splitKFoldStratified(X, y):
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













































        
    
  

    

