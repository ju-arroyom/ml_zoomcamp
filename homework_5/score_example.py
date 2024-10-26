import requests
## a new customer informations
client1 = {"job": "student", "duration": 280, "poutcome": "failure"}
client2 = {"job": "management", "duration": 400, "poutcome": "success"}


def score_client(client):
    url = 'http://localhost:8787/predict' ## this is the route we made for prediction
    response = requests.post(url, json=client) ## post the customer information in json format
    result = response.json() ## get the server response
    print(result)

for i, client in enumerate([client1, client2]):
    print(f"Scoring client {i}")
    score_client(client)