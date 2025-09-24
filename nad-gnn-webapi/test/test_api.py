import requests
import json

API_URL = "http://172.16.0.1:5000/predict"
def test_predict():
    # Example features
    sample_features = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]

    response = requests.post(
        API_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps({"features": sample_features})
    )
    if response.status_code == 200:
        result = response.json()
        print("Prediction output:", result["prediction"])
    else:
        print(f"Request failed with status code {response.status_code}: {response.text}")

if __name__ == "__main__":
    test_predict()
