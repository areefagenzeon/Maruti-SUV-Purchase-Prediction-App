import pickle
#from ml_model import sst
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler
import pandas as pd
#model.pkl - trained ml model

#Desirilize - read the binary file - trained ML model
clf=pickle.load(open('model.pkl','rb'))

####################################################################################################
#for getting range decided on xtrain - repeat the ML steps till Normalization again
import pandas as pd
df=pd.read_csv("SUV_Purchase.csv")
#step 2 feature Engineering - drop unecessay or unimportant features - simplifying the dataset
df=df.drop(['User ID','Gender'],axis =1)#axis=1 i.e columns  ....axis =0 i.e rows

#step 3- loading the data
#setting the data into input and output values
X=df.iloc[:,:-1].values #iloc==>index location 2D array
Y=df.iloc[:,-1:].values #2D array

#step 4 - Split dataset into training in test
#Training and Testing the dataset
#more data-Trainig; Less data-Testing datai.e Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
#####################################################################################################
app=Flask(__name__)

@app.route("/") #annotation triggers the methods following - default annotation that renders the 1st web page to the browser
def hello():
    return render_template('index.html')

#jinja2 - template engine - which would be going to templates folder and selecting the webpage - hence folder name should be templates

@app.route('/predict',methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()]) #POST is working fine - we are getting i/ps from frontend
    features=[int(x) for x in request.form.values()]
    print(features)  #[20,30000] - [1]
    sst=StandardScaler().fit(X_train) #range would be decided on Xtrain - carry forwarded for prediction
    output=clf.predict(sst.transform([features]))
    print(output) #[0]
    if output[0] == 0:
        return render_template('index.html',pred=f'The Person will not be able to purchase the SUV')
    else:
        return render_template('index.html',pred=f'The Person will be able to purchase the SUV')


if __name__ == "__main__":
    app.run(debug=True)
