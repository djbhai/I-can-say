# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 03:21:54 2018

@author: Dheeraj
"""
import mysql.connector

from flask import Flask, render_template,request,jsonify

app = Flask(__name__)
sentence_to_test=''


@app.route('/')
def menu():
 
   
 return  render_template('homepage.html')
    


#main menu
@app.route('/words')
def selection_table():
 words=[]
 data=[]
 model_data=[]   
 cnx = mysql.connector.connect(user='root',password='',host='127.0.0.1', database='words')
 cursor=cnx.cursor()
 query=("SELECT `word` FROM `content` ORDER BY RAND() LIMIT 10")
 cursor.execute(query)
 
 
 
 for word in cursor:
     
     data.append( str( word[0]))
     
     
     words.append(word)
     
     
 for(item)   in data:
     
     item1=item.replace('\r\n',"")
     item2=item1.replace("'","" )
     model_data.append(item2)
 
 print(model_data)
 
 cursor.close()
 cnx.close()

     

 
 return render_template('getwords.html',words=model_data)

@app.route('/learning',methods=['GET'])
def learning():
 cnx=mysql.connector.connect(user='root',password='',host='127.0.0.1',database='words')
 cursor=cnx.cursor()
    
    
    
 a = request.args.get('a')          #the word pronounced by the user
    
    
 b=request.args.get('b')          #the actual word 
 b=b.strip()

 
 
 
 
 cursor.execute("UPDATE content SET number_of_attempts=number_of_attempts+1 WHERE word='%s'" %(b))
 cnx.commit()
 

    
 if(a==b):
     
     cursor.execute("UPDATE content SET correct_responses=correct_responses+1 WHERE word='%s'" %(b))
     cnx.commit()
     
        
    
 cnx.close()
 
 #the machine learning code can go here
 
 
 
 return jsonify(a=a,b=b)       #the values may need to updated if the machine learning algorithm recognises the word
        

@app.route('/phrases')

def phrases():
 words=[]
 data=[]
 model_data=[]   
 cnx = mysql.connector.connect(user='root',password='',host='127.0.0.1', database='phrases')
 cursor=cnx.cursor()
 query=("SELECT `sentence` FROM `phrase` ORDER BY RAND() LIMIT 10")
 cursor.execute(query)
 
 
 
 for word in cursor:
     
     data.append( str( word[0]))
     
     
     words.append(word)
     
     
 for(item)   in data:
     
     item1=item.replace('\r\n',"")
     item2=item1.replace("'","" )
     model_data.append(item2)
 
 print(model_data)
 
 cursor.close()
 cnx.close()


 return render_template('getphrases.html',words=model_data)

    


    


       
    

#once the selection_button is pressed:
#@app.route('/', methods=['POST'])
#def selection_result():
# sentence_to_test=request.selection['selection_table']
# return render_template('play_and_record.html',sentence=sentence_to_test)

#once the stop_test button is pressed:
#@app.route('/playrecord', methods=['POST'])
#def stoprecord():
# voice = request.form['raw_voice']
# sentence_google=request.form['google_translation']
# sentence_from_ml=ml_model(voice)
# score_,sentence_to_test=result_cal(sentence,sentence_google,sentence_from_ml)
 #return render_template('show_result.html', score=score_)

#once the test_remaining button is pressed:
#@app.route('/testremain', methods=['POST'])
#def testremain():
#return render_template('play_and_record.html',sentence=sentence_to_test)

if __name__ == '__main__':
 app.run(debug=True)
