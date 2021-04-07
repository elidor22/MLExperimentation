from flask import Flask, jsonify, request, render_template, make_response, redirect
app = Flask(__name__)
from regressor import Predictor

pred = Predictor()
@app.route("/", methods=["GET", "POST"])
def get_and_return():
    if request.method == "POST":
        #id = int(request.form['id'])
        #burst_time_raw = request.form['burst_time']
        val1= request.json['val1']
        val2= request.json['val2']
        val3= request.json['val3']
        val4= request.json['val4']
        val5= request.json['val5']
        val6= request.json['val6']
        val7= request.json['val7']
        val8= request.json['val8']
    
    res = pred.predict(val1,val2,val3,val4,val5,val6,val7,val8)
    
    response = make_response(
        jsonify({
            'Prediction: ':round(float(res),6),
            
        })

    )
    return response
if __name__ == '__main__':
    app.run(debug=True)