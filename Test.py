# Importing everything
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# Reding data
data = pd.read_csv("student-mat.csv", sep=";")
# Setting the data to only the one we want
data = data[["G1", "G2", "G3", "study_time", "failures", "absences"]]

# Saying what we want to predict/our output
predict = "G3"
# Setting X to the data without answers...
X = np.array(data.drop(columns=[predict]))
# ...and Y to the answer we want to get.
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# This is used to save the most accurate model
'''
best = 0
for _ in range(1000):
    # Splitting the data to Train/Test
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

    # Setting up the training model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    # Setting up the Test
    acc = linear.score(x_test, y_test)
    print("accuracy: ", acc)

    if acc > best:
        # Saving model (wb mode makes it create the file if it doesn't already exist(f is name of file))
        with open("student_model.pickle", "wb") as f:
            pickle.dump(linear, f)

'''
pickle_in = open("student_model.pickle", "rb")
linear = pickle.load(pickle_in)

# Writing the gradient + y intercept(= line of best fit)
print("Co: ", linear.coef_)
print("Intercept: ", linear.intercept_)

# Writing the predicting, data and answer for every student
predictions = linear.predict(x_test)
# This for statement is going till range = len( len is equals to the full length of predictions)
# Here is the documentation for this for loop:
# https://www.w3schools.com/python/python_for_loops.asp
# https://www.w3schools.com/python/ref_func_range.asp
# https://realpython.com/len-python-function/
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Plotting information
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
