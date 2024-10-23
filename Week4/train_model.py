import pickle
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

iris = datasets.load_iris()
X = iris.data  
y = iris.target  

model = DecisionTreeClassifier()
model.fit(X, y)

with open('iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as iris_model.pkl")
