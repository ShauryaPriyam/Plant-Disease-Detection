import uuid
from pathlib import Path
from flask import Flask, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import torch
import pandas as pd
from werkzeug.utils import secure_filename


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)

model.load_state_dict(
    torch.load(
        "plant_disease_model_1_latest.pt",
        map_location=torch.device("cpu"),
        weights_only=True
    )
)

model.eval()

def prediction(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))

    input_data = TF.to_tensor(image)
    input_data = input_data.unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(input_data), dim=1)[0]

    scores, indices = torch.topk(probabilities, k=3)
    return [
        {"index": int(index), "confidence": float(score)}
        for score, index in zip(scores, indices)
    ]


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
HEALTHY_CLASSES = {3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38}


def assessment_guidance(prediction_index, confidence):
    """Return a cautious image-screening priority and general IPM next steps."""
    if prediction_index == 4:
        return (
            "Needs a clearer image", "uncertain",
            "The image may not show a recognizable leaf. Retake a close, well-lit photo of the affected plant part.",
            ["Photograph one affected leaf against a plain background.",
             "Check several plants and note whether symptoms are spreading.",
             "Ask a local extension worker to review persistent or worsening symptoms."]
        )

    if prediction_index in HEALTHY_CLASSES:
        if confidence >= 0.70:
            return (
                "No clear disease pattern detected", "low",
                "The image screening currently leans toward a healthy leaf. Continue routine scouting; this does not rule out other problems.",
                ["Inspect plants regularly, including the underside of leaves.",
                 "Keep tools clean and avoid moving from affected plants to healthy plants without cleaning.",
                 "Use balanced crop care and avoid unnecessary pesticide applications."]
            )
        return (
            "Uncertain result — check again", "uncertain",
            "The model is not confident enough to rule out a crop-health issue. Compare with other plants and consider expert review.",
            ["Take another clear photo in natural light.",
             "Mark and recheck the plant over the next few days.",
             "Seek local agricultural advice if symptoms worsen or appear on more plants."]
        )

    if confidence >= 0.75:
        level, label = "high", "High concern — inspect promptly"
    elif confidence >= 0.45:
        level, label = "moderate", "Possible issue — monitor closely"
    else:
        level, label = "uncertain", "Uncertain result — check again"

    name = str(disease_info['disease_name'][prediction_index]).lower()
    steps = [
        "Inspect nearby plants and the undersides of leaves; note how many plants show similar symptoms.",
        "Remove fallen or clearly affected plant material where practical, and clean tools before using them elsewhere.",
        "Avoid overhead watering when foliage disease is suspected; maintain airflow and keep a record of changes.",
        "Confirm the diagnosis with a local agricultural extension worker before applying any crop-protection product."
    ]
    if "virus" in name or "curl" in name:
        steps[1] = "Avoid moving potentially affected seedlings or plant material between plots; clean hands and tools after handling."
        steps[2] = "Check for insect vectors and seek local extension guidance on appropriate integrated control."
    elif "mite" in name or "spider" in name:
        steps[1] = "Check leaf undersides for mites and webbing; where suitable, use a gentle water spray to dislodge pests."
        steps[2] = "Conserve beneficial insects and avoid broad-spectrum sprays that can disrupt natural enemies."

    return (
        label, level,
        "This is an image-based screening priority, not a forecast of field spread or a confirmed diagnosis. Crop, stage and location are recorded as context and do not change the CNN score.",
        steps
    )

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('assessment.html', crops=list_crop_options())

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files.get('image')
        if not image or not image.filename:
            return render_template('assessment.html', error="Choose or capture a leaf photo before analysing.",
                                   crops=list_crop_options()), 400
        extension = Path(secure_filename(image.filename)).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return render_template('assessment.html', error="Use a JPG, PNG or WebP image.",
                                   crops=list_crop_options()), 400

        upload_dir = Path(app.root_path) / 'static' / 'uploads'
        upload_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}{extension}"
        file_path = upload_dir / filename
        try:
            image.save(file_path)
            # Decode once up front so invalid files produce a friendly form error.
            with Image.open(file_path) as uploaded:
                uploaded.verify()
            top_predictions = prediction(str(file_path))
        except Exception:
            file_path.unlink(missing_ok=True)
            return render_template('assessment.html', error="We could not read that image. Please choose another photo.",
                                   crops=list_crop_options()), 400

        top = top_predictions[0]
        pred = top['index']
        title = str(disease_info['disease_name'][pred])
        recommendations, risk_level, risk_note, ipm_steps = assessment_guidance(pred, top['confidence'])
        ranked = [{
            "title": str(disease_info['disease_name'][item['index']]),
            "confidence": item['confidence'] * 100,
            "index": item['index']
        } for item in top_predictions]
        selected_crop = request.form.get('crop', '').strip()
        predicted_crop = title.split(' : ', 1)[0].strip().lower()
        crop_aliases = {"pepper, bell": "pepper", "pepper bell": "pepper"}
        normalized_selected_crop = crop_aliases.get(selected_crop.lower(), selected_crop.lower())
        normalized_predicted_crop = crop_aliases.get(predicted_crop, predicted_crop)
        crop_mismatch = bool(selected_crop and selected_crop.lower() != 'other'
                             and normalized_predicted_crop not in
                             ('background_without_leaves', 'background')
                             and normalized_selected_crop != normalized_predicted_crop)
        return render_template(
            'results.html', title=title, desc=str(disease_info['description'][pred]),
            top_predictions=ranked, confidence=top['confidence'] * 100,
            image_url=f"/static/uploads/{filename}", pred=pred,
            crop=selected_crop, growth_stage=request.form.get('growth_stage', '').strip(),
            location=request.form.get('location', '').strip(), crop_mismatch=crop_mismatch,
            risk_title=recommendations, risk_level=risk_level, risk_note=risk_note,
            ipm_steps=ipm_steps
        )


def list_crop_options():
    return ["Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach",
            "Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato", "Other"]

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
