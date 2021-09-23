from flask import Flask,render_template,request
import numpy as np
import pickle
import re 
import string
import pandas as pd
app = Flask(__name__)
vector = pickle.load(open('vectrorization','rb'))
model = pickle.load(open('Fake_news_model','rb'))


 #df1 = pd.read_csv('corpus.csv')  

#corpus1 =[]
#for i in range(len(df1)):
 #   review = re.sub('a-zA-Z'," ",df1['0'][i])

  #  corpus1.append(review) 
def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ", text)
    text = re.sub('https?://\S+!www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text 
@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST',"GET"])
def predict():
    nt = request.form.get('new_text')

    new_text=nt
    testing_news = {"text":[new_text]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(word_drop) 
    new_x_test = new_def_test["text"]
    new_xv_test = vector.transform(new_x_test)
    pred_LR = model.predict(new_xv_test)
    
     
    if(pred_LR[0]):
        return render_template('result.html', pred="True News")
    else:
         return render_template('result.html', pred1="Fake News ") 

   
    
if __name__ == '__main__':
    app.run(debug=True)