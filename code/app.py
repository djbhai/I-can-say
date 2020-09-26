from flask import Flask, redirect, url_for, request, render_template
import numpy as np
import threading

app = Flask(__name__)

total_score={}
chosen_sentence={}
remain_sentence={}


class myThread (threading.Thread):  
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):                   
        modeltraininginthread()
 

def modeltraininginthread():
    print "model_training_finished"


def calc(thisvoice,thissentence,chosen_sentence):
	return 'the',1


@app.route('/', methods=['GET'])
def showselection():
	return render_template('showselection.html')

@app.route('/', methods=['POST'])
def getselection():
	chosen_sentence[1]=request.form['Sentence']
	remain_sentence[1]=chosen_sentence[1]
	total_score[chosen_sentence[1]]=0
	return redirect(url_for('play',sentence=chosen_sentence[1]))


@app.route('/play/<sentence>', methods=['GET'])
def play(sentence):

	'''
	thread1 = myThread()
	thread1.start()
	print "model_training"
	'''
	return render_template('playandrecord.html',play_sentence=sentence,addr="/play/"+sentence)


@app.route('/play/<sentence>', methods=['POST'])
def playtest(sentence):
	thissentence=sentence
	thisvoice='dog'
	remain_sentence[1],thisscore=calc(thisvoice,thissentence,chosen_sentence[1])
	total_score[chosen_sentence[1]] = thisscore + total_score[chosen_sentence[1]]
	return redirect(url_for('result',score='80'))


@app.route('/result/<score>', methods=['GET'])
def result(score):
	return render_template('result.html',myscore=score,addr="/play/"+remain_sentence[1])


if __name__ == '__main__':
	app.run()
