import os
import requests
import io
import tempfile  # ← Asegúrate de importar esto
from flask import Flask, request, jsonify
from google.cloud import vision

# Configurar credenciales de Google Cloud
# Cargar credenciales desde la variable de entorno
google_credentials = os.getenv("GOOGLE_CREDENTIALS_JSON")

if google_credentials:
    # Crear un archivo temporal con las credenciales
    temp_credentials = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_credentials.write(google_credentials.encode())
    temp_credentials.close()

    # Configurar la variable de entorno para Google Cloud
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_credentials.name

# Inicializar el cliente de Google Vision
client = vision.ImageAnnotatorClient()

# Inicializar Flask
app = Flask(__name__)

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