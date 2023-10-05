# Name : Harshit
# SRN : PES1UG20CS161

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

# Read the excel file and drop all the rows with atleast one null value
dataset = pd.read_excel('file.xlsx')
dataset = dataset.dropna()

# Customers who have made a transaction between 1st Dec 2010 to 31st Aug 2011
training_data = dataset[(dataset['InvoiceDate'] >= '2010-12-01') & (dataset['InvoiceDate'] <= '2011-08-31')]

# Customers who have not made any subsequent purchase between Sep 2011 and Dec 2011
temp0_data = dataset[(dataset['InvoiceDate'] >= '2011-09-01') & (dataset['InvoiceDate'] <= '2011-12-31')]
temp0_customer_ids = list(set(list(temp0_data['CustomerID'])))

# Create the labels for the customers from training_data
target_values = []
for customer_id in training_data['CustomerID']:
    if customer_id not in temp0_customer_ids:
        target_values.append('Churn')
    else:
        target_values.append('Not Churn')
training_data['Target'] = target_values

# Filter the attributes required for training the models
training_features = training_data[['Quantity', 'UnitPrice', 'Country', 'Target']]

# Assign labels to values from attributes 'Quantity' and 'UnitPrice'
temp0 = list(training_features['UnitPrice'].values)
mid0 = sum(temp0) / len(temp0)

temp1 = [i for i in temp0 if i < mid0]
temp2 = [i for i in temp0 if i >= mid0]
mid1 = sum(temp1) / len(temp1)
mid2 = sum(temp2) / len(temp2)

new_unitPrice = []
for unitPrice in temp0:
    if unitPrice < mid1:
        new_unitPrice.append('low')
    elif unitPrice > mid1 and unitPrice < mid2:
        new_unitPrice.append('medium')
    else:
        new_unitPrice.append('high')

temp0 = list(training_features['Quantity'].values)
mid0 = sum(temp0) / len(temp0)

temp1 = [i for i in temp0 if i < mid0]
temp2 = [i for i in temp0 if i >= mid0]
mid1 = sum(temp1) / len(temp1)
mid2 = sum(temp2) / len(temp2)

new_quantity = []
for quantity in temp0:
    if quantity < mid1:
        new_quantity.append('low')
    elif quantity > mid1 and unitPrice < mid2:
        new_quantity.append('medium')
    else:
        new_quantity.append('high')

training_features['UnitPrice'] = new_unitPrice
training_features['Quantity'] = new_quantity

# Save the dataframe 'training_features' as csv file
training_features.to_csv('data.csv')

df = pd.read_csv('data.csv')
df = df.drop(['Unnamed: 0'], axis=1)

train_size = int(len(df) * 0.80)

train = df[:train_size]
test = df[train_size:]

# Function for calculating the entropy of the entire or subset of the dataframe
def get_entropy_of_dataset(df):
    labels = df.iloc[:,-1].values
    distinct_labels = {}

    for label in labels:
        if label not in distinct_labels:
            distinct_labels[label] = 1
        else:
            distinct_labels[label] += 1

    entropy = 0
    length = len(labels)
    for distinct_label in distinct_labels:
        pi = distinct_labels[distinct_label] / length
        entropy += (-pi) * np.log2(pi)

    return entropy

# Function for calculating the average information of a specific attribute
def get_avg_info_of_attribute(df, attribute):
    attribute_values = list(set(df[attribute].values))
    total_length = len(df)
    average_info = 0

    for attribute_value in attribute_values:
        temp_df = df[df[attribute] == attribute_value]
        entropy = get_entropy_of_dataset(temp_df)
        average_info += ((len(temp_df) / total_length) * entropy)

    return average_info

# Function for calculating the Information gain of a specific attribute
def get_information_gain(df, attribute):
    return get_entropy_of_dataset(df) - get_avg_info_of_attribute(df, attribute)

# Function for selecting an attribute with highest information gain
def get_selected_attribute(df):
    attributes = list(df.columns[:-1].values)
    igs = []
    
    for attribute in attributes:
        igs.append(get_information_gain(df, attribute))

    max_index = igs.index(max(igs))
    return attributes[max_index]

# Function for creating the tree (Tree is created using the datatype 'dict')
def make_tree(df):
    if get_entropy_of_dataset(df) == 0:
        return ''.join(list(set(df.iloc[:,-1].values)))
    if len(df.columns.values) == 1:
        values = list(df.iloc[:,-1].values)
        return ''.join(list(max(values, key= lambda x : values.count(x))))
    if df.empty:
        return []
    tree = {}
    attribute = get_selected_attribute(df)
    attribute_values = list(set(df[attribute].values))
    
    tree[attribute] = {}
    for attribute_value in attribute_values:
        temp_df = df[df[attribute] == attribute_value]
        temp_df = temp_df.drop([attribute], axis = 1)
        tree[attribute][attribute_value] = make_tree(temp_df)

    return tree

# Function for predicting the label of a given instance using the tree created.
def pred(tree, instance):
    sub_tree = tree
    while type(sub_tree) != str:
        attribute = list(sub_tree.keys())[0]
        attribute_value = instance[attribute]
        sub_tree = sub_tree[attribute][attribute_value]

    return sub_tree

# Function for evaluating the accuracy of the decision tree.
def evaluate(tree, test):
    length = len(test)
    correct = 0

    for i in range(length):
        prediction = pred(tree, test.iloc[i])
        if prediction == test.iloc[i, -1]:
            correct += 1

    return (correct / length) * 100

# Create the tree
tree = make_tree(train)

# Evaluate the tree
tree_accuracy = evaluate(tree, test)

# Training the ANN model
training_features = training_data[['Quantity', 'UnitPrice', 'Country', 'Target']]

le0 = LabelEncoder()
training_features['Country'] = le0.fit_transform(training_features['Country'])
le1 = LabelEncoder()
training_features['Target'] = le1.fit_transform(training_features['Target'])

x = training_features.iloc[:,:-1].values
y = training_features.iloc[:,-1].values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

ann_model = Sequential()
ann_model.add(Dense(4, activation='relu'))
ann_model.add(Dense(10, activation='relu'))
ann_model.add(Dense(1, activation='sigmoid'))

ann_model.compile(loss = 'mse', optimizer = 'adam', metrics=['accuracy'])
ann_model.fit(xtrain, ytrain, epochs = 1, batch_size = 1, verbose = 1)

y_pred = ann_model.predict(xtest)
correct = 0
for i in range(len(y_pred)):
    if y_pred[i] > 0.50:
        y_pred[i] = 1
    else:
        y_pred[i] = 0
for i in range(len(y_pred)):
    if y_pred[i] == ytest[i]:
        correct += 1

ann_accuracy = (correct / len(y_pred)) * 100

# Train the SVM model
svm_model = SVC(kernel='rbf')
svm_model.fit(xtrain, ytrain)

svm_accuracy = 100 * accuracy_score(ytest, svm_model.predict(xtest))

print('------------------')
print('------------------')
print('Decision Tree (ID3) Accuracy = {}'.format(tree_accuracy))
print('ANN Model Accuracy = {}'.format(ann_accuracy))
print('SVM Model Accuracy = {}'.format(svm_accuracy))
print('------------------')
print('------------------')
