from flask import Flask, render_template, url_for, make_response,jsonify,request
import tensorflow_hub as hub
import numpy as np
import threading


app = Flask(__name__,template_folder='templates')


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=500)])
  except RuntimeError as e:
    print(e)

def semantic(search1,search2):
    comparison = model([search1,search2])
    return np.inner(comparison[0],comparison[1])

def task():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    model = hub.load(module_url)
    tf.keras.backend.clear_session()


@app.route('/')
def menu():
    threading.Thread(target=task).start()
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
    


