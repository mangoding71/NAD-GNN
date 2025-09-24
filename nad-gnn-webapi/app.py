from flask import Flask, request, jsonify
import torch
from model import NAD_GNN

app = Flask(__name__)

# Load
try:
    model = torch.jit.load("nadgnn_model.pt")
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("features", None)
        if data is None:
            return jsonify({"error": "Missing 'features' in request body"}), 400

        tensor_data = torch.tensor(data).float()

        with torch.no_grad():
            output = model(tensor_data)

        return jsonify({"prediction": output.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    #select your ip and port, 172.16.0.1 is the default IP address of our host.
    app.run(host="172.16.0.1", port=5000, debug=True)
