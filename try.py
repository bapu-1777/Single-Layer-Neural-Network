# Single Layer Neural Network

import numpy as np

np.random.seed(2)
rg = np.random.default_rng()
bais=0
l_rate = 0.1
epochs = 10
epochs_loss=[]

def generate_data(n_features, n_values):

    x_train = rg.random((n_features,n_values))
    weights = rg.random((1,n_values))[0]
    y_train=np.random.choice([0,1],n_features)
    return x_train,y_train,weights

def get_weighted_sum(feature, weights, bais):
    return np.dot(feature,weights)+bais

def hard_limit(w_sum):
    if w_sum>=0:
        return 1
    else:
        return 0


def loss(target, prediction):
    return prediction-target

def update_weight(weight, l_rate, target, prediction, feature):
    new_weight = []
    for x,w in zip(feature,weight):
        new_w = w+ l_rate*(target-prediction)*x
        new_weight.append(new_w)

    return new_weight



x_train, y_train, weigts = generate_data(3, 4)
for i in range(epochs):
    individual_loss=[]
    for i in range(len(x_train)):
        feature = x_train[i]
        target = y_train[i]
        w_sum = get_weighted_sum(feature, weigts, bais)
        prediction = hard_limit(w_sum)
        F_loss = loss(target, prediction)
        individual_loss.append(F_loss)
        weigts = update_weight(weigts, l_rate, target, prediction, feature)

    average_loss = sum(individual_loss)/len(individual_loss)
    epochs_loss.append(average_loss)
    print(epochs_loss)