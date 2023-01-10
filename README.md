# Self-Diagnosis App
The aim of this project is for people with no knowledge of medicine to be able to diagnose themselves to a certain extent, hence making them aware of the diseases they might be exposed to. This project takes the first step in producing training data which can then be used to predict diseases. 

If this problem is solved, it can help people to start the treatment early, as time is extremely important especially with diseases which have a lengthy period of restoration. An additional benefit of this problem being solved is the chances of a misdiagnosis would reduce considerably. 

## Design
* Uses the [Diseases Database](https://www.kaggle.com/datasets/hagari/disease-and-their-symptoms) which includes approximately 800 diseases with about 80 symptoms per disease on average.
* Uses the [pandas](https://pandas.pydata.org/) library for file-handling. 
* Implements relevant plots and figures using the [matplotlib](https://matplotlib.org/) library.
* Applies data mining algorithms using the [scikit-learn](https://scikit-learn.org/stable/) library.  

## Functionality
* Choose from 2 upto 10 symptoms from given list of more than 300 possible symptoms.
* Accuracy of diagnosis increases with more number of symptoms.
* Get the most probable disease with >95% accuracy. 

## Accuracy of algorithms used:

| Algorithm  | Accuracy |
| ------------- | ------------- |
| K-Nearest Neighbor  | 92.89%  |
| Bernoulli Bayes  | 96.33%  |
| Logistic Regression  | 97.62%  |



