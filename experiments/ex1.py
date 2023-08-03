def scripting(i):
    pyTitle = f'fullex_G{i}'
    pyString ="""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import multiprocessing

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import ExtraTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm.classes import OneClassSVM
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier    
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import MultinomialNB  
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action='ignore')
def get_data(sample_size = 1000,n_noisy = 100, n_clusters=2, shift=1, shuffle=False):
    shift_map = {1:(1,0),2:(1,1),3:(0,1),4:(0,2),5:(1,2),6:(2,2),7:(2,1),8:(2,0)}
    import random
    FileName = '../data/unified_data_cluster1.csv'
    use_columns = [0,1] + [i+2 for i in range(n_noisy)] + [10002]
    columns = ['signal1', 'signal2'] + ['noisy'+str(i+3) for i in range(n_noisy)] + ['label']
    
    n = 10000 # sample size of each big cluster file
    skip = sorted(random.sample(range(0,n),n-sample_size))
    for i in range(n_clusters):
        if i == 0:
            df = pd.read_csv(f'../data/multicluster/unified_normal_cluster{i+1}.csv',
                             header=None,
                              skiprows= skip,
                             usecols=use_columns, 
                             names=columns)
        else:
            shift1 = shift*shift_map[i][0]
            shift2 = shift*shift_map[i][1]
            df1 = pd.read_csv(f'../data/multicluster/unified_normal_cluster{i+1}.csv',
                             header=None,
                              skiprows= skip,
                             usecols=use_columns, 
                             names=columns)
            df1['signal1'] = df1['signal1'].apply(lambda x: x+shift1)
            df1['signal2'] = df1['signal2'].apply(lambda x: x+shift2)
            df = pd.concat([df,df1])
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return(df)

def model_selection(i):
    sample_size, n_noisy,n_clusters,shift=i
    weight=None
    classifiers = {'ExtraTreeClassifier': ExtraTreeClassifier(class_weight=weight),
             'DecisionTreeClassifier': DecisionTreeClassifier(class_weight=weight),
             'MLPClassifier': MLPClassifier(),
             'KNeighborsClassifier': KNeighborsClassifier(),
             'SGDClassifier': SGDClassifier(class_weight=weight),
             'RidgeClassifierCV': RidgeClassifierCV(class_weight=weight),
             'RidgeClassifier': RidgeClassifier(class_weight=weight),
             'PassiveAggressiveClassifier': PassiveAggressiveClassifier(class_weight=weight), 
             'GaussianProcessClassifier': GaussianProcessClassifier(),
             'AdaBoostClassifier': AdaBoostClassifier(),
             'GradientBoostingClassifier': GradientBoostingClassifier(),
             'BaggingClassifier': BaggingClassifier(),
             'RandomForestClassifier': RandomForestClassifier(class_weight=weight),
             'BernoulliNB': BernoulliNB(),
             'CalibratedClassifierCV': CalibratedClassifierCV(),
             'GaussianNB': GaussianNB(),
             'LabelPropagation': LabelPropagation(),
             'LabelSpreading':  LabelSpreading(),
             'LinearDiscriminantAnalysis': LinearDiscriminantAnalysis(),
             'LinearSVC': LinearSVC(class_weight=weight),
             'LogisticRegression':LogisticRegression(class_weight=weight),
             'LogisticRegressionCV': LogisticRegressionCV(class_weight=weight),
             'NearestCentroid': NearestCentroid(),
             'NuSVC': NuSVC(class_weight=weight),
             'Perceptron': Perceptron(class_weight=weight),
             'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
             'SVC': SVC(class_weight=weight)}
    df = get_data(sample_size=sample_size, n_noisy = n_noisy, n_clusters=n_clusters,shift=shift, shuffle=True)
    X, y = df.loc[:, df.columns != 'label'].to_numpy(),df.label.to_numpy()
    cv_scores = []
    for name, clf in classifiers.items():
        cv_scores.append(cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs =-1))
        clf.fit(X, y)

    print([sample_size, n_noisy,n_clusters,shift]+ cv_scores)
    return([sample_size, n_noisy,n_clusters,shift]+ cv_scores)

def main(i):
    repeats = 3
    n_noisy_list = [0,2,10,50,7,90,100,200,400,600,800,1000,5000,8000]
    n_cluster_list = [i for i in range(2,10)]
    n_sample_list = [100, 1000]
    shift_list = [-0.5,-0.2,0,0.2,0.5,1,2,4]
    paras = list(itertools.product(n_sample_list, n_noisy_list, n_cluster_list,shift_list))
    chunk=paras[i::100]
    with multiprocessing.Pool(10) as pool:
        result = pool.map(model_selection,chunk*repeats)
    df_cvs = pd.DataFrame(result,columns=['sample_size', 'n_noisy','n_clusters', 'shift']+list(classifiers.keys()))    
    df_cvs.to_pickle('./data/multi_clusters_algorithms_compare_full.pkl')
if __name__ == '__main__': 
    main(#)""".replace('main(#)',f'main({i})').replace('multi_clusters_algorithms_compare_full',f'multi_clusters_algorithms_compare_full_{i}')
    
    f = open(f"{pyTitle}.py", "w")
    f.write(pyString)
    f.close()
    
    shString = f"""#!/bin/bash
#SBATCH -A p30656                # Allocation
#SBATCH -p long                # Queue
#SBATCH -t 168:00:00             # Walltime/duration of the job
#SBATCH -N 1                 # Number of Nodes
#SBATCH --mem=64G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=10    # Number of Cores (Processors)
#SBATCH --mail-user=stringlwh@gmail.com  # Designate email address for job communications
#SBATCH --mail-type=END     # Events options are job BEGIN, END, NONE, FAIL, REQUEUE
#SBATCH --output=.    # Path for output must already exist
#SBATCH --error=.     # Path for errors must already exist
#SBATCH --job-name="{pyTitle}"       # Name of job

# load modules you need to use
module load python/anaconda3
source activate ailimits

# A command you actually want to execute:
python3 {pyTitle}.py"""
        
    f = open(f"{pyTitle}.sh", "w")
    f.write(shString)
    f.close()
    return f"{pyTitle}.sh"

import subprocess
for i in range(0, 100):
    job = scripting(i)
    subprocess.run(["chmod","u+x",job])
    subprocess.run(["qsub",job])
    print(f'running job {job}')