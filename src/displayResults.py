class visualize:
  
    def visualizeConfusionMatrix(true_value, predicted_value):
        from sklearn.metrics import confusion_matrix
        # from sklearn.metrics import plot_confusion_matrix
        confusion_m = confusion_matrix(true_value, predicted_value)

        import matplotlib.pyplot as plt
        from mlxtend.plotting import plot_confusion_matrix
        import numpy as np

        fig, ax = plot_confusion_matrix(conf_mat=confusion_m, figsize=(2, 2), cmap="OrRd")
        plt.show()

            
    def showScores(true_value, predicted_value):
        from sklearn.metrics import precision_recall_fscore_support as score
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        precision, recall, fscore, support = score(true_value, predicted_value)
        

        print('accuracy: {}'.format(accuracy_score(true_value, predicted_value)))
        print('precision: {}'.format(precision_score(true_value, predicted_value)))

        print('recall: {}'.format(recall_score(true_value, predicted_value)))
        print('fscore: {}'.format(f1_score(true_value, predicted_value)))
        print('support: {}'.format(support))