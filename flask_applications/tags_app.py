"""
GET : Sends data in simple or unencrypted form to the server.
HEAD : Sends data in simple or unencrypted form to the server without body. 
PUT : Replaces target resource with the updated content.
DELETE : Deletes target resource provided as URL.
"""

from flask import Flask, redirect, url_for, request
from functions import transform_dl_fct, feature_USE_fct
import pickle

app = Flask(__name__)


def load():
    """fonction qui charge le modèle entrainé, l'explainer, le dataset sur lequel va porter l'api et le détail des features
    utilisés par le modèle"""
    name_suff = 'use'
    # Read dictionary pkl file
    with open('ovr_sgdc_{}.pkl'.format(name_suff), 'rb') as fp:
        return pickle.load(fp)


def pred_fct(model, sentence):
    """Predict the tags of a sentence
    """

    prep_sentence = [transform_dl_fct(sentence, html=True)]

    b_size = 1

    feature_sentence = feature_USE_fct(prep_sentence, b_size)

    predict_phrase = model.predict(feature_sentence)

    mlb = pickle.load(open('mlb_binarizer.pkl', 'rb'))

    tags_predict = mlb.inverse_transform(predict_phrase)[0]

    print(tags_predict)

    return tags_predict

model = load()


@app.route('/')
def hello_world():
    return 'Bienvenue sur l\'API du Projet 4 !'


@app.route('/success/<sentence>')
def success(sentence):
    return str(pred_fct(model, sentence))


@app.route('/predict_tags', methods=['POST', 'GET'])
def predict_tags():
    if request.method == 'POST':
        sentence = request.args.get('sentence')
        return {'response' : pred_fct(model, sentence)}  #redirect(url_for('success', sentence=sentence))

    else:
        sentence = request.args.get('sentence')
        return redirect(url_for('success', sentence=sentence))


# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run(debug=True)
