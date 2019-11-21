#version 0.1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve

def model_performance(algorithm, training_x,testing_x, training_y,testing_y, cols, cf):
    algorithm.fit(training_x, training_y)
    predictions = algorithm.predict(testing_x)
    probabilities = algorithm.predict_proba(testing_x)
    
    if cf == 'coefficients':
        coefficients = pd.DataFrame(algorithm.coef_.ravel())
    elif cf == 'features':
        coefficients = pd.DataFrame(algorithm.feature_importances_)
    columns_df = pd.DataFrame(cols)
    coef_summry = pd.merge(coefficients, columns_df,left_index = True, right_index = True, how = 'left')
    
    #Confusion Matrix
    conf_matrix = confusion_matrix(testing_y, predictions)
    fig, ax = plt.subplots(1, 3, figsize =(18,5))
    sns.heatmap(conf_matrix,square = True, annot = True, fmt = '.0f', ax = ax[0])
    ax[0].set_title('Confusion Matrix')
    
    #ROC Curve
    y_score = log.decision_function(testing_x)
    fpr, tpr, _ = roc_curve(testing_y, y_score)
    roc_auc = auc(fpr,tpr)
    ax[1].plot(fpr, tpr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc))
    ax[1].set_xlim([-0.01, 1.00])
    ax[1].set_ylim([-0.01, 1.01])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('ROC Curve')
    ax[1].legend(loc='lower right')
    ax[1].plot([0, 1], [0, 1], color='red', lw=1, linestyle='--')
    ax[1].set_aspect('equal')
    
    #Precision Recall Curve
    precision, recall, thresholds = precision_recall_curve(testing_y, y_score)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]
    ax[2].plot(precision, recall, label='Precision-Recall Curve')
    ax[2].plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    ax[2].set_title('Precision-Recall Curve')
    ax[2].set_xlabel('Precision')
    ax[2].set_ylabel('Recall')
    ax[2].set_aspect('equal')
    ax[2].set_xlim([0.0,1.01])
    ax[2].set_ylim([0.0,1.01])
    plt.show()
    
    #feature importances
    coef_sumry    = (pd.merge(coefficients,column_df,left_index= True,right_index= True, how = "left"))
    coef_sumry.columns = ["coefficients", "features"]
    coef_sumry = coef_sumry.sort_values(by = "coefficients", ascending = False)
    plt.figure(figsize = (20,5.5))
    ax = plt.gca()
    ax.grid(color='gray', linestyle='dashed')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.bar(coef_sumry["features"], coef_sumry["coefficients"])
    x = plt.gca().xaxis

    for item in x.get_ticklabels():
        item.set_rotation(90)

