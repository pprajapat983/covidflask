from flask import Flask,render_template
app=Flask(__name__)
import pickle
#import requests
from flask import request

file=open('corona.pkl','rb')
clf=pickle.load(file)
file.close()
@app.route('/',methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        myDict=request.form
        fever=int(myDict['fever'])
        age=int(myDict['age'])
        pain=int(myDict['pain'])
        runnynose=int(myDict['runnynose'])
        breathing=int(myDict['breathing'])
        input_features=[fever,pain,age,runnynose,breathing]
        infProb=clf.predict_proba([input_features])[0][1]
        print(infProb)
        return render_template('show.html',inf=round(infProb*100))
    return render_template('index1.html')
    #return 'Hello World' + str(infProb)


if __name__ == '__main__':
    app.run(debug=True)