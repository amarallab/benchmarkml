
def scripting(i):
    pyTitle = f'RESAMPLED80_CELL_E{i}'
    pyString ="""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import multiprocessing

from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split,cross_val_score,cross_validate
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
             'Perceptron': Perceptron(class_weight=weight),
             'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
             'SVC': SVC(class_weight=weight)}
             
def get_data(n_noisy = 100, shuffle=False):
    import random
    FileName = '../cell_data/pbmc_final.csv'
    use_columns = [1,2] + [i+3 for i in range(n_noisy)] + [8407]
    columns = ['signal1', 'signal2'] + ['noisy'+str(i+3) for i in range(n_noisy)] + ['label']
    df = pd.read_csv(FileName,
                header=0,
                usecols=use_columns,
                names=columns)
      
    dfs = []
    for i in [0,1,2]:
        d = df[df.label==i]
        d = d.sample(n=48)
        dfs.append(d)
    df = pd.concat(dfs)
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    return(df)

def model_selection(i):
    sample_size = 80*3
    n_clusters = 3
    shift = 0
    n_noisy = i
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
             'Perceptron': Perceptron(class_weight=weight),
             'QuadraticDiscriminantAnalysis': QuadraticDiscriminantAnalysis(),
             'SVC': SVC(class_weight=weight)}
    df = get_data(n_noisy = n_noisy, shuffle=True)
    X, y = df.loc[:, df.columns != 'label'].to_numpy(),df.label.to_numpy()
    X = np.log(X + 1)
    cv_scores = []
    scoring = {'accuracy': 'accuracy',
              'balanced_accuracy': 'balanced_accuracy'}
    for name, clf in classifiers.items():
        cv_scores.append(cross_validate(clf, X, y, cv=5, scoring=scoring, n_jobs =-1))
        clf.fit(X, y)

    print([sample_size, n_noisy,n_clusters,shift]+ cv_scores)
    return([sample_size, n_noisy,n_clusters,shift]+ cv_scores)

def main(i):
    n_noisy_list = [0,2,10,50,100,1000,5000,8000]
    repeats = 20
    paras = n_noisy_list * repeats
    chunk=paras[i::8]
    with multiprocessing.Pool(10) as pool:
        result = pool.map(model_selection, chunk)

if __name__ == '__main__':
    main(#)""".replace('main(#)',f'main({i})')

    f = open(f"{pyTitle}.py", "w")
    f.write(pyString)
    f.close()

    shString = f"""#!/bin/bash
#SBATCH -A b1022                # Allocation
#SBATCH -p buyin                # Queue
#SBATCH -t 48:00:00             # Walltime/duration of the job
#SBATCH -N 1                 # Number of Nodes
#SBATCH --mem=64G               # Memory per node in GB needed for a job. Also see --mem-per-cpu
#SBATCH --ntasks-per-node=4    # Number of Cores (Processors)
#SBATCH --mail-type=END     # Events options are job BEGIN, END, NONE, FAIL, REQUEUE
#SBATCH --mail-user=stringlwh@gmail.com  # Designate email address for job communications
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
for i in range(0, 8):
    job = scripting(i)
    subprocess.run(["chmod","u+x",job])
    subprocess.run(["qsub",job])
    print(f'running job {job}')
