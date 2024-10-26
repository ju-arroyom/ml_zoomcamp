import pickle
from flask import Flask, request, jsonify

client = {"job": "management", "duration": 400, "poutcome": "success"}

def load_artifacts(path):
    with open(path, 'rb') as f_in:
        artifact = pickle.load(f_in)
    return artifact


def predict_client(dv, model, client):
    X = dv.transform([client])  ## apply the one-hot encoding feature to the customer data 
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred[0]


dv = load_artifacts("dv.bin")
model = load_artifacts("model2.bin")
app = Flask('hw5')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    prediction = predict_client(dv, model, client)
    decision = prediction >= 0.5
    
    result = {
        'probability': float(prediction),
        'decision': bool(decision),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8787)