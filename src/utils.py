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

            X = []
            for i in range(0, len(df)):
                yorum = re.sub("[^AaBbCcÇçDdEeFfGgĞğHhİiIıJjKkLlMmNnOoÖöPpRrSsŞşTtUuÜüVvYyZz']", ' ', df['text'][i])
                yorum = re.sub("[']", '', yorum) #drop things that without letters
                yorum = yorum.lower()
                yorum = yorum.strip()
                yorum = yorum.split()

                yorum = [stemmer.stem(word) for word in yorum if word not in stop_word_list]
                yorum = ' '.join(yorum)

                X.append(yorum)


            pickle.dump(X, open("normalization_X_data.pickle", "wb"))

        y = df['target'].to_numpy()
    
        return X,y
    
        
    
  

    

