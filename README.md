# Clustering-Classification-WeatherData
 Application of Clustering (Gaussian Mixture and Mean-Shift), Classification (SVM and Naive Bayes) techniques for Weather Prediction

## Introduction
 I have used weather data of the Indian capital Delh i. It contains 14 different columns of which a few are object type and the others are integer or float data. Columns named condition and wind direction are string data type whereas the other columns like temperature, windgust, rain etc have numeric data. I have implemented the following clustering and classification techniques for this dataset:
i.	Gaussian Mixture Clustering
ii.	Mean-Shift Clustering
iii. Support Vector Machines Classification
iv.	Naïve Bayes Classification

## Result of Clutering:
### Gaussian Mixture Clustering
![Gaussian Mixture](https://user-images.githubusercontent.com/31332352/103679784-4c8b2780-4f53-11eb-8e66-99623c253ca0.png)
### Mean Shift Clustering
![Mean Shift](https://user-images.githubusercontent.com/31332352/103679791-4d23be00-4f53-11eb-817e-20fba4180ab2.png)


## Result of Classification:

### SVM Confusion Matrix and FScore
![CMatrix_SVM](https://user-images.githubusercontent.com/31332352/103680074-a55ac000-4f53-11eb-9e1e-4bb7514fb5a9.png)
![SVM_FScore](https://user-images.githubusercontent.com/31332352/103680072-a55ac000-4f53-11eb-921f-cdddd2f50369.png)
### Naive Bayes Confusion Matrix and FScore
![CMatrix_NB](https://user-images.githubusercontent.com/31332352/103680281-f10d6980-4f53-11eb-9490-fceb0321a55c.png)
![NaiveBayes_FScore](https://user-images.githubusercontent.com/31332352/103680282-f1a60000-4f53-11eb-84e9-898888b0bb56.png)

## Conclusion:
I would like to point out a few things for clustering:
i.	It is easy to implement Gaussian Mixture than Mean Shift as it requires minimum data wrangling.
ii.	The output of Gaussian Mixture is not 100% correct but is correct upto 80% which means that it can be applied given the lesser amount of work it needs.
iii. Both Gaussian Mixture and Mean Shift do not over-cluster the data (in general cases).

The conclusion of classification is as follows:
The SVM model consumed a couple of seconds more to generate the results when compared to the Naïve Bayes. Moreover, the Naïve Bayes model treats each attribute as independent before processing which does not fit the condition for application in weather as each attribute results in an affect over the other. For example, if the temperature increases, the condition is expected to be dry or humid with extreme heat. However, SVM considers the interaction of each attributes with the other upto a certain extent before predicting. This shows that the SVM has better application with the dataset whose attributes are inter-related.

