from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pylab as plt
import numpy as np

X,y=load_iris(return_X_y=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

pipe=Pipeline([

    ("scaling", StandardScaler()),
    ("model", KNeighborsClassifier())
])

# print(pipe.get_params()) this gets me the parameters for the model and scaler

mod=GridSearchCV(estimator=pipe,param_grid={'model__n_neighbors':[1,2,3,4,5,6,7,8,9,10]}, cv=3)
mod.fit(X_train,y_train)


prediction=mod.predict(X_test)

acc=accuracy_score(prediction,y_test)
print(f"Accuracy Score:{acc*100}%")

conf=confusion_matrix(prediction,y_test)
print(f"Confusion Matrix:\n{conf}")

plt.scatter(prediction + np.random.normal(0,0.03,size=len(prediction)), y_test + np.random.normal(0,0.03,size=len(y_test)))
plt.xlabel("Prediction")
plt.ylabel("Actual")
plt.title("Model Predictions vs. Model Accuracy")
plt.show()

print("Best # of Neighbours for KNeighborsClassifier")
pd.DataFrame(mod.cv_results_) #this shows me a table using pandas which compares the cv results across neighbours 1-10

