# 🌿 AI-Assisted Crop Disease Detection

### Deep Learning-Based Plant Disease Identification Using Convolutional Neural Networks (CNN)

An AI-powered crop-health screening application designed to help farmers and agricultural users identify plant diseases from leaf images. The system uses a PyTorch-based Convolutional Neural Network (CNN) trained on the PlantVillage dataset to classify leaf images into **39 different disease and healthy-plant categories**.

The project aims to support early disease identification, improve crop-health awareness and encourage timely agricultural intervention.

---

## 1. Project Overview

Crop diseases and pest infestations are major threats to agricultural productivity, farmer income and food security. Early identification is important because delayed recognition can allow infections to spread and increase crop losses.

This project provides an image-based disease screening solution through a web application. Users can select a crop, upload an image of an affected leaf and receive an AI-generated prediction with a confidence score.

The project is designed as an initial component of a broader AI-assisted crop-health decision-support platform.

### Key Features

- **39-Class Disease Classification:** Identifies plant diseases and healthy conditions across 39 categories.
- **CNN-Based Image Recognition:** Uses a deep learning model implemented with PyTorch.
- **Image-Based Diagnosis:** Upload a leaf image to obtain a predicted disease category.
- **Confidence Score:** Displays the model's prediction confidence.
- **Alternative Predictions:** Shows other possible classification matches.
- **Disease Information:** Provides descriptions to help users understand predicted conditions.
- **IPM Guidance:** Offers general Integrated Pest Management (IPM) recommendations.
- **Supplement Information:** Includes a supplementary information section for agricultural users.
- **Web-Based Interface:** Provides an accessible interface for interacting with the trained model.

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.10 |
| Deep Learning Framework | PyTorch |
| Model Architecture | Convolutional Neural Network (CNN) |
| Training Dataset | PlantVillage |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Image Processing | PIL / OpenCV (as used in the application) |
| Data Management | CSV files |
| Model Format | PyTorch `.pt` |

---

## 3. Project Structure

The repository is organised into the following directories:

```text
Plant-Disease-Detection/
│
├── demo_images/
│   └── Application demonstration images
│
├── Flask Deployed App/
│   │
│   ├── static/
│   │   └── CSS, JavaScript and static assets
│   │
│   ├── templates/
│   │   └── HTML templates
│   │
│   ├── venv/
│   │   └── Python virtual environment (local)
│   │
│   ├── app.py
│   │   └── Flask application entry point
│   │
│   ├── CNN.py
│   │   └── CNN model architecture
│   │
│   ├── disease_info.csv
│   │   └── Disease information
│   │
│   ├── supplement_info.csv
│   │   └── Supplement information
│   │
│   ├── plant_disease_model_1_latest.pt
│   │   └── Trained PyTorch model
│   │
│   ├── requirements.txt
│   │   └── Python dependencies
│   │
│   ├── Procfile
│   │
│   └── Readme.md
│
├── Model/
│   └── Model training and experimentation files
│
├── test_images/
│   └── Sample images for testing
│
└── README.md
```

---

## 4. Dataset and Model

### Dataset

The project uses the **PlantVillage dataset** for training the plant disease classification model.

The dataset contains labelled images of healthy and diseased plant leaves and provides the foundation for supervised image classification.

### Model Architecture

A Convolutional Neural Network (CNN) is used to learn visual patterns from leaf images and classify them into different categories.

The model is implemented using PyTorch.

**Model capabilities:**

- Extracts visual features from leaf images.
- Learns disease-specific patterns during training.
- Classifies images into 39 categories.
- Produces prediction scores for possible classes.

### Model File

The trained model is included in the repository:

```text
Flask Deployed App/
    plant_disease_model_1_latest.pt
```

No separate Google Drive download is required when the repository already contains this file.

---

## 5. Running the Project Locally

Follow these steps to run the Flask application on your local machine.

### Prerequisites

<mark>Python 3.10 is strictly required.</mark>

- Python 3.10
- pip
- Git (if cloning the repository)

**Important:** Use Python 3.10 for this project. Other Python versions are not the supported environment for the provided setup instructions.

### Step 1: Open the Project Directory

Open your terminal in the root project directory.

Navigate to the Flask application folder:

```bash
cd "Flask Deployed App"
```

### Step 2: Create a Virtual Environment

Create a new Python virtual environment using Python 3.10:

```bash
py -3.10 -m venv venv
```

### Step 3: Activate the Virtual Environment

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

After activation, your terminal should display `(venv)`.

### Step 4: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 5: Run the Flask Application

Start the application:

```bash
py app.py
```

### Step 6: Open the Application

Once the Flask server starts successfully, open the local URL displayed in your terminal, usually:

```text
http://127.0.0.1:5000/
```

The application should now be accessible through your browser.

---

## 6. Testing the Application

The repository includes sample images in the `test_images/` directory.

These images can be used to test the disease-classification functionality without uploading your own photographs.

### Testing Workflow

1. Start the Flask application.
2. Open the application in your browser.
3. Navigate to the AI Engine.
4. Select the appropriate crop.
5. Upload a leaf image.
6. View the predicted disease category and confidence score.
7. Review alternative predictions and available disease information.

The predictions should be treated as AI-generated screening results rather than guaranteed agricultural diagnoses.

---

## 7. Current Prototype and Future Development

The current implementation establishes the image-based disease-screening component of a broader crop-health decision-support platform.

The proposed expansion focuses on integrating multiple sources of agricultural information into a unified system.

| Development Stage | Planned Capabilities |
|---|---|
| Current Prototype | CNN-based disease classification and web application |
| Short-Term | Controlled pilot using selected crops and expert-labelled cases |
| Medium-Term | Punjab-based field validation, weather-risk integration and regional deployment |
| Long-Term | Multi-state expansion and an India-wide crop-health surveillance network |

### Proposed Future Enhancements

- Weather-based disease-risk assessment.
- Crop-growth-stage-aware risk analysis.
- Geo-tagged farmer reports and GIS-based disease hotspot mapping.
- Expert validation and referral for uncertain predictions.
- Integration of optional temperature, humidity and leaf-wetness sensors.
- Multilingual advisories in Hindi, Punjabi and English.
- Improved Integrated Pest Management (IPM) recommendations.
- Continuous learning using expert-confirmed field observations.

**Important:** These are proposed development directions and are not all implemented in the current Flask prototype.

---

## 8. Project Vision

The long-term objective is to move crop-health management from isolated disease identification towards a continuous, integrated decision-support process:

**Detection → Risk Assessment → Advisory → Expert Validation → Surveillance → Learning**

By combining deep learning, environmental information, agricultural expertise and geospatial intelligence, the proposed platform aims to support earlier intervention, more informed crop-management decisions and improved agricultural productivity.

---

## 9. Limitations and Responsible Use

- Model predictions depend on image quality and similarity to the training dataset.
- Real-world field conditions may differ from curated training images.
- Visually similar diseases may lead to incorrect predictions.
- A high confidence score does not guarantee a correct diagnosis.
- Agricultural treatment decisions should be confirmed using locally validated guidance or an agricultural expert.

The application is intended as a decision-support tool, not a replacement for professional agricultural diagnosis.

---

## 10. Acknowledgement

Developed as part of the project **AI-Assisted Early Detection and Management of Crop Diseases and Pest Infestations**.

**Group 20 | Indian Institute of Technology Ropar**

Problem Statement ID: 26131

**Project Objective:** To develop an integrated, farmer- and extension-worker-friendly crop-health decision-support platform using artificial intelligence and agricultural intelligence.

---

*Early Detection. Better Decisions. Healthier Crops.* 🌱
