import os
import json
import io
import tempfile
import requests
from flask import Flask, request, jsonify
from google.cloud import vision

# Load Google Cloud credentials from environment variable
google_credentials = os.getenv("GOOGLE_CREDENTIALS_JSON")

if google_credentials:
    temp_credentials = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    temp_credentials.write(google_credentials.encode())
    temp_credentials.close()
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_credentials.name

# Initialize Flask and Google Vision client
app = Flask(__name__)
client = vision.ImageAnnotatorClient()

# Activity classification lists
EXTREME_SPORTS = {
    "Skydiving", "Bungee jumping", "Rock climbing", "Motocross", "Surfing", "Snowboarding",
    "Skateboarding", "Ski", "Ski Equipment", "Winter sports"
}
NORMAL_ACTIVITIES = {"Walking", "Running", "Cycling", "Swimming", "Gym", "Yoga"}

def classify_activity(labels):
    """Classifies activity based on detected labels and assigns a risk level."""
    detected_activities = [label["description"] for label in labels]

    # Check for extreme sports
    extreme_detected = EXTREME_SPORTS.intersection(detected_activities)

    if extreme_detected:
        if "Ski" in detected_activities or "Snowboarding" in detected_activities:
            return "Extreme sport detected: Ski/Snowboarding", 7  # Moderate-high risk
        return "Extreme sport detected", 9  # High risk
    else:
        return "Normal activity", 2  # Low risk

@app.route("/analyze", methods=["POST"])
def analyze_image():
    """Receives an image URL, analyzes it using Google Vision API, and classifies the activity"""
    data = request.get_json()
    
    if "image_url" not in data:
        return jsonify({"error": "You must provide an image URL"}), 400

    image_url = data["image_url"]

    try:
        # Download the image
        response = requests.get(image_url, stream=True)
        if response.status_code != 200:
            return jsonify({"error": "Failed to download the image"}), 400

        # Read the image into memory
        image_content = io.BytesIO(response.content)
        image = vision.Image(content=image_content.read())

        # Analyze the image using Google Vision AI
        response = client.label_detection(image=image)
        labels = [{"description": label.description, "score": label.score} for label in response.label_annotations]

        # Classify the activity and assign a risk level
        activity, risk = classify_activity(labels)

        return jsonify({
            "detected_labels": labels,
            "detected_activity": activity,
            "risk_level": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)