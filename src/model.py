import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# read CSV file
data = pd.read_csv('data/heart.csv')

# normalize data
scaler = preprocessing.StandardScaler()
x = data[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']].values
x = scaler.fit(x).transform(x.astype(float))
y = data['output'].values

# split test data and train data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# test the diffrent K : 1 to 10
for n in range(1,11):
    test_model = KNeighborsClassifier(n).fit(x_train, y_train)
    print(metrics.accuracy_score(y_test, test_model.predict(x_test)))
    print(f'_____________________________________________________________________{n}')

# fit model
model = KNeighborsClassifier(10).fit(x_train, y_train)

# print F1_score 
print('\n\n________________________________________________________________________')
print(f' F1_score : {metrics.f1_score(y_test, model.predict(x_test))}')
print('________________________________________________________________________')

# show the true and predict Y
print('predict VS real:')
print(f'predict :  {model.predict(x_test)}')
print(f'real : {y_test}')
