#importing libraries
import random
import numpy as np
import pandas as pd
import datetime as dt
from matplotlib import pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift, estimate_bandwidth

#classification libraries
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix

plt.style.use('seaborn-darkgrid')


#importing dataset
data = pd.read_csv(r'weather_data.csv')

# selecting only required columns
data = data.loc[1:20000,['datetime_utc', ' _conds', ' _dewptm', ' _fog', ' _hail', ' _hum', ' _pressurem', ' _rain', ' _snow',' _tempm', ' _thunder', ' _tornado', ' _vism',' _wdire']]

#renaming columns to convert into understandable names
weather_df = data.rename(index=str, columns={' _conds': 'condition', ' _dewptm':'dew', ' _fog':'fog',' _hail':'hail', ' _hum': 'humidity', ' _pressurem': 'pressure',' _rain':'rain',' _snow':'snow',' _tempm': 'temperature', ' _thunder':'thunder', ' _tornado': 'tornado', ' _vism':'visibility',' _wdire':'wind_direction'})

#replacing NaN with previous valid values
weather_df.ffill(inplace=True)
weather_df.dropna()

#sampling the dataset to 2000 random samples
weather_df = weather_df.sample(n=2000)
weather_df = weather_df.drop('datetime_utc',axis=1)

print('The weather dataset used here contains',weather_df.shape[0],'rows and',weather_df.shape[1],'columns')

print('____________________Clustering 1: GAUSSIAN MIXTURE_________________')

clustering = weather_df[['humidity','temperature']].copy()

# Gaussian Mixture

encoder = OrdinalEncoder()

clustering_transformed = encoder.fit_transform(clustering)

X=clustering_transformed

model = GaussianMixture(n_components=7)

yhat = model.fit_predict(X)

clusters = np.unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = np.where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(X[row_ix, 0], X[row_ix, 1],label=cluster)
# show the plot
plt.title('Gaussian Clustering of Humidity and Temprature')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.legend()
plt.show()


print('____________________Clustering 2: MEAN SHIFT CLUSTERING_________________')

# Mean Shfit

# Estimate bandwith
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=60)

# Fit Mean Shift with Scikit
meanshift = MeanShift(bandwidth=bandwidth)
meanshift.fit(X)
labels = meanshift.labels_
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

#  Predict the cluster for all the samples
P = meanshift.predict(X)

# Generate scatter plot for training data
colors = list(map(lambda x: 'r' if x == 1 else 'g' if x == 2 else 'b' if x == 3 else 'c' if x == 4 else 'm' if x == 5 else 'y' if x == 6 else '#2f4f4f', P))
plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)
plt.title(f'Mean Shift Clustering with {n_clusters_} estimated clusters')
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.show()

print('____________________Classification 1: SVM_________________')

classification_data = weather_df.copy()
temp_label = ['very cold','cold','temperate','Warm','Hot']
classification_data['temperature_n'] = pd.qcut(classification_data['temperature'], q=5,labels=temp_label)

lab_enc = preprocessing.LabelEncoder()
classification_data['temperature_n'] = lab_enc.fit_transform(classification_data['temperature_n'])

#SVM
svm_x = classification_data.drop('temperature_n',axis=1)._get_numeric_data()
#imputing missing values with mean
svm_x.fillna(svm_x.mean(), inplace = True)
svm_y = classification_data['temperature_n']
svm_y.astype('int')
scaler = StandardScaler()
scaler.fit(svm_x)
svm_x = scaler.transform(svm_x)
svm_x_train, svm_x_test, svm_y_train, svm_y_test = train_test_split(svm_x, svm_y, test_size=0.2, random_state=1)
svc=SVC(kernel='rbf') #Default hyperparameters
svc.fit(svm_x_train, svm_y_train)
svm_y_pred=svc.predict(svm_x_test)
print('Accuracy Score:')
print(metrics.accuracy_score(svm_y_test,svm_y_pred))
print(classification_report(svm_y_test,svm_y_pred))
print('The Confusion Matrix for SVM is: \n')
print(confusion_matrix(svm_y_test,svm_y_pred))

#Plot Confusion Matrix
plot_confusion_matrix(svc,svm_x_test,svm_y_test,display_labels=temp_label,cmap='Blues')
plt.title('Confusion Matrix of SVM')
#plt.savefig('SVM_Confusion.png')
plt.show()

#Predicted Values to 
ax1=plt.hist(svm_y_test,histtype='step',color='r',label='actual value')
ax2=plt.hist(svm_y_pred,histtype='step',color='blue',label='predicted value')
plt.title('Predicted and Actual Values of SVM')
plt.legend()
plt.show()
#plt.savefig('SVM_graph.png')


print('____________________Classification 2: NAIVE BAYES_________________')

lab_enc = preprocessing.LabelEncoder()
classification_data['temperature_n'] = lab_enc.fit_transform(classification_data['temperature_n'])

y = classification_data['temperature_n']
X = classification_data.drop('temperature_n',axis=1)._get_numeric_data()


X.fillna(X.mean(), inplace = True)
y = classification_data['temperature_n']
y.astype('int')
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)

#Gaussian Naive Bayes classifier

nb = GaussianNB()
nb.fit(X_train,y_train)
nbpred = nb.predict(X_test)
nb_conf_matrix = confusion_matrix(y_test, nbpred)
nb_acc_score = accuracy_score(y_test, nbpred)
#print("confussion matrix")
#print(nb_conf_matrix)
print("\n")
print("Accuracy of Naive Bayes model:",nb_acc_score*100,'\n')
print(classification_report(y_test,nbpred))

#Plot Confusion Matrix
plot_confusion_matrix(nb,X,y,display_labels=temp_label,cmap='Blues')
plt.title('Confusion Matrix of Naive Bayes')
#plt.savefig('SVM_Confusion.png')
plt.show()