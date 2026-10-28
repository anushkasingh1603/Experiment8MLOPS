import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

model=joblib.load("model.pkl")

testdf=pd.read_csv("Iris - Iris.csv")

testdf=testdf.drop("Id", axis=1)

X=testdf.drop("Species", axis=1)
y=testdf["Species"]
pred=model.predict(X)
acc=accuracy_score(pred , y)
print(acc)
assert len(pred) == len(y) #test cases checking if the length of predictions and actual labels are the same

assert acc>=0.90 #acc test case checking if the accuracy is greater than 1.0 which is not possible, so this will fail
print("All test cases passed and model is ready to deploy!")
