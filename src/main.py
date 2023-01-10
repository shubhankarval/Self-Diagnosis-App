import random
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression


# Removing irrelevant information from the file (every column after symptoms)
# Since we can see that "symptoms" is a "list" using brackets, we can simply remove everything that shows up after "]"

# only keep the name and symptoms columns
def cleaning():
    f = open('disease_and_symptoms.txt', 'r', encoding='utf-8')
    w = open('disease_and_symptoms2.txt', 'w', encoding='utf-8')
    f.readline()
    for line in f:
        idx1 = line.index(",")
        idx2 = line.index("]")
        temp = line[idx1 + 1:idx2]
        w.write(temp)
        w.write("\n")
    


# Next, turn this "list" into readable data. The way it's currently formatted, it's symptoms: symptom name,
# followed by symptoms: appearance rate given disease.

# First, get all the diseases as a key, with the values being lists containing symptom name and the appearance rate.

diseases = {}


def create_dict():
    f = open('disease_and_symptoms2.txt', 'r', encoding='utf-8')
    for line in f:
        idx = line.index(",")
        disease = line[:idx]
        diseases[disease] = []
        symptoms = line[idx + 1:]
        lst = symptoms.split('{"symptoms":"')
        lst = lst[1:]
        for i in range(len(lst)):
            lst[i] = lst[i][:-3]
        for i in range(0, len(lst), 2):
            diseases[disease].append(lst[i:i + 2])




# Next, we create a list containing every symptom that exists.
# The inputs for the program will be "1"s and "0"s depending on whether or not the symptom is present.

symptoms = []


def create_list():
    for v in diseases.values():
        for symptom in v:
            if symptom[0] not in symptoms:
                symptoms.append(symptom[0])



# Now, we can create some simulated data. For every disease, we'll create, let's say, 100 entries for now.
# Then, we can split it into 80/20 for training/testing.

# The way the entry will be created is using the random library, and rolling the appearance rate out of 100
# to generate either a 1 or 0. If an entry contains only 1 symptom, it will be removed to ensure some level of accuracy.


def create_data():
    train = open('training.csv', 'w', encoding='utf-8')
    test = open('testing.csv', 'w', encoding='utf-8')
    for i in range(len(symptoms)):
        # Some symptoms contain commas in them, so we need to remove it for the csv columns to stay consistent
        symptoms[i] = symptoms[i].replace(",", "")
        train.write(symptoms[i])
        test.write(symptoms[i])
        train.write(",")
        test.write(",")
    train.write("Disease\n")
    test.write("Disease\n")
    for k, v in diseases.items():
        temp_symptoms = []
        temp_appearance = []
        for symptom in v:
            temp_symptoms.append(symptom[0])
            temp_appearance.append(symptom[1])
        counter = 0
        while counter < 100:
            lst = []
            for i in range(len(symptoms)):
                if symptoms[i] in temp_symptoms:
                    idx = temp_symptoms.index(symptoms[i])
                    temp = random.randint(1, 100)
                    if temp <= int(temp_appearance[idx]):
                        lst.append(1)
                    else:
                        lst.append(0)
                else:
                    lst.append(0)
            if lst.count(1) > 1 and counter < 80:
                counter += 1
                lst.append(k)
                lst = map(str, lst)
                lst = ",".join(lst)
                train.write(lst)
                train.write("\n")
            elif lst.count(1) > 1 and counter < 100:
                counter += 1
                lst.append(k)
                lst = map(str, lst)
                lst = ",".join(lst)
                test.write(lst)
                test.write("\n")


# Finally, we can input this data into an algorithm.
# We'll use k-NN, Bernoulli Bayes, and Logistic Regression

def kNN():
    training = pd.read_csv('training.csv')
    testing = pd.read_csv('testing.csv')

    trainingX = training.loc[:, training.columns != "Disease"]
    trainingY = training.loc[:, training.columns == "Disease"]

    testingX = testing.loc[:, training.columns != "Disease"]
    testingY = testing.loc[:, training.columns == "Disease"]

    scores = []
    r = range(1, 10)
    for i in r:
        classifier = KNeighborsClassifier(n_neighbors=i)
        classifier.fit(trainingX, trainingY)
        predictions = classifier.predict(testingX)
        scores.append(metrics.accuracy_score(testingY, predictions))
    print(scores)
    plt.plot(r, scores)
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.show()


def bernoulli_Bayes():
    training = pd.read_csv('training.csv')
    testing = pd.read_csv('testing.csv')

    trainingX = training.loc[:, training.columns != "Disease"]
    trainingY = training.loc[:, training.columns == "Disease"]

    testingX = testing.loc[:, training.columns != "Disease"]
    testingY = testing.loc[:, training.columns == "Disease"]

    bnnb = BernoulliNB()
    bnnb.fit(trainingX, trainingY)

    predictions = bnnb.predict(testingX)
    print(metrics.accuracy_score(testingY, predictions))


def logRegression():
    training = pd.read_csv('training.csv')
    testing = pd.read_csv('testing.csv')

    trainingX = training.loc[:, training.columns != "Disease"]
    trainingY = training.loc[:, training.columns == "Disease"]

    testingX = testing.loc[:, training.columns != "Disease"]
    testingY = testing.loc[:, training.columns == "Disease"]

    lr = LogisticRegression()
    lr.fit(trainingX, trainingY)

    predictions = lr.predict(testingX)
    print(metrics.accuracy_score(testingY, predictions))


# Note that create_data() can take a while, and will produce a new simulated dataset each time
# Make sure to run all three at once

# cleaning()
create_dict()
create_list()
print(diseases)
print("\n")
print(symptoms)
# create_data()


# You can run the following algorithms without having to run the create_ functions each time.
# Only do so if you wish to try a new dataset.


# Accuracy of 0.458856783919598 with k = 7 (will vary for different datasets)
# If you run kNN, pyplot should return a graph for k values and their accuracy
# This will take a while to complete
# kNN()

# Accuracy of 0.6550251256281407 (will vary for different datasets)
# bernoulli_Bayes()

# Accuracy of 0.6474874371859296 (will vary for different datasets)
# This will take a while to complete
# logRegression()


