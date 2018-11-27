# exploring features in the subset of Titanic data, and deploying decision tree to predict the survival rate

%matplotlib inline
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read and print basic info

data = pd.read_csv(os.path.join('classification/titanic-subset', 'titanic.csv'), sep = ',')
data.head(10)

# observe the distribution of data

data['survived'].value_counts(normalize = True)
sns.countplot(data['survived'])

# observe the relation between class and survival rate

sns.countplot(data['pclass'], hue = data['survived'])

# observe the relation between title and survival rate

data['title'] = data['name'].apply(lambda x: x.split(',')[1] if len(x.split(',')) > 1 else x.split(',')[0].apply(lambda x: x.split()[0]))
data['title'].value_counts()
data['survived'].groupby(data['title']).mean()

# observe the relation between gender and survival rate

data['sex'].value_counts(normalize = True)
data['survived'].groupby(data['sex']).mean()

# observe the relation between board place and survival rate

data['embarked'].value_counts()
data['survived'].groupby(data['embarked']).mean()

#############################################################################################

# feature transform

def name(data):
    data['len'] = data['name'].apply(lambda x: len(x))
    data['title'] = data['name'].apply(lambda x: x.split('.')[1] if len(x.split(',')) > 1 else x.split(',')[0]).apply(lambda x: x.split()[0])
    del data['name']

    return data

def age(data):
    data['age_flag'] = data['age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    grouped_age = data.groupby(['title', 'pclass'])['age']
    data['age'] = grouped_age.transform(lambda x: x.fillna(data['age'].mean()) if pd.isnull(x.mean()) else x.fillna(x.mean()))

    return data

def embark(data):
    data['embarked'] = data['embarked'].fillna('Southampton')

    return data

def dummies(data, columns=['pclass','title','embarked', 'sex']):
    for i in columns:
        data[i] = data[i].apply(lambda x: str(x))
        new_cols = [i + '_' + j for j in data[i].unique()]
        data = pd.concat([data, pd.get_dummies(data[i], prefix = i)[new_cols]], axis = 1)
        del data[i]

    return data

# preprocess data : transform some charactor and drop off redundant features

drop_col = ['row.names', 'home.dest', 'room', 'ticket', 'boat']
data = data.drop(drop_col, axis = 1)
data.head()

data = name(data)
data = age(data)
data = embark(data)
data = dummies(data)
data.head()

# call decision tree and prdeict result

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

train_x, test_x, train_y, test_y = train_test_split(data.iloc[:,1:], data.iloc[:,0], test_size = 0.2, random_state = 30)
model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 4, min_samples_leaf = 5)
model.fit(train_x, train_y)

def measure_performance(x, y, z, show_accuracy = True, show_classification_report = True, show_confusion_matrix = True):
    pred = z.predict(x)
    if show_accuracy:
        print('Accuracy:{0:.3f}'.format(metrics.accuracy_score(y, pred)), '\n')

    if show_classification_report:
        print('Classification Report')
        print(metrics.show_classification_report(y, pred), '\n')

    if show_confusion_matrix:
        print('Confusion Matrix')
        print(metrics.show_confusion_matrix(y, pred), '\n')

measure_performance(test_x, test_y, model)

# visualize decision tree

import graphviz

tree_data = tree.export_graphviz(model, out_file = None, feature_names = train_x.columns)
scrach = graphviz.Source(tree_data)
