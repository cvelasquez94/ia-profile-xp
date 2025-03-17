import os
import requests
import io
from flask import Flask, request, jsonify
from google.cloud import vision

# Configurar credenciales de Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"

# Inicializar Flask
app = Flask(__name__)

# Cliente de Google Vision AI
client = vision.ImageAnnotatorClient()

# Headers con User-Agent para evitar bloqueos
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

@app.route("/analyze", methods=["POST"])
def analyze_image():
    """Recibe una URL de imagen, la descarga, la analiza con Google Vision AI y devuelve etiquetas"""
    data = request.get_json()
    
    if "image_url" not in data:
        return jsonify({"error": "Debes proporcionar una URL de imagen"}), 400

    image_url = data["image_url"]

    try:
        # Descargar la imagen con headers para evitar bloqueos
        response = requests.get(image_url, headers=HEADERS, stream=True)
        
        if response.status_code != 200:
            return jsonify({"error": "No se pudo descargar la imagen"}), 400

        # Leer la imagen en memoria
        image_content = io.BytesIO(response.content)

        # Crear imagen para Google Vision AI
        image = vision.Image(content=image_content.read())

        # Analizar imagen con Vision AI
        response = client.label_detection(image=image)
        labels = response.label_annotations

        results = [{"description": label.description, "score": label.score} for label in labels]
        
        return jsonify({"labels": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)