from flask import Flask, render_template, url_for, make_response,jsonify,request
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np


app = Flask(__name__)

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)



def semantic(search1,search2):
    comparison = model([search1,search2])
    return np.inner(comparison[0],comparison[1])

@app.route('/')
def menu():
    return render_template("index.html")

@app.route('/<search1>/<search2>',methods=['POST','GET'])
def deploy(search1,search2):
    compare = semantic(search1,search2)
    compare = compare*100
    compare = str(compare)
    compare = compare.strip("")
    response = {
        "Semantic Similarity": compare
    }
    if request.method == 'POST':
        return make_response(jsonify(response),200)
    else:
        return render_template("results.html",compare=compare,)
    

if __name__ == "main":
    app.run(port=int(os.environ.get("PORT", 5000)), host='0.0.0.0')