# Chest X-Ray Pneumonia Classifier — Web App

A deployed inference service for the [pneumonia classifier](https://github.com/Mubydeji/pneumonia-classifier)
trained in a companion notebook. FastAPI backend, static HTML/CSS/JS frontend,
containerized and deployed to Google Cloud Run. Model weights are hosted on
Hugging Face Hub and downloaded at container startup, rather than bundled
into the Docker image or committed to this repo.

**Live demo:** _add your Cloud Run URL here once deployed_

## Architecture

```
Browser (static frontend)
        │  POST /predict (image file)
        ▼
FastAPI app (Cloud Run container)
        │  downloads weights from Hugging Face Hub at startup
        ▼
ResNet-50 (CPU inference)
        │
        ▼
{ probability, threshold, label }
```

## Running locally

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

export MODEL_REPO_ID="muby26/chest-xray-pneumonia-resnet50"
export MODEL_FILENAME="resnet50_pneumonia.pt"

uvicorn app.main:app --reload
```

Visit `http://localhost:8000`.

## Running with Docker

```bash
docker build -t pneumonia-app .
docker run -p 8080:8080 \
  -e MODEL_REPO_ID="muby26/chest-xray-pneumonia-resnet50" \
  -e MODEL_FILENAME="resnet50_pneumonia.pt" \
  pneumonia-app
```

Visit `http://localhost:8080`.

## Deploying to Google Cloud Run

```bash
gcloud run deploy pneumonia-classifier \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_REPO_ID=muby26/chest-xray-pneumonia-resnet50,MODEL_FILENAME=resnet50_pneumonia.pt
```

## Limitations

This is a research prototype, not a validated clinical tool. See the
[training repository](https://github.com/Mubydeji/pneumonia-classifier)
for the full methodology, evaluation, and stated limitations, including
the fact that the model was trained and tested on a single-institution
dataset and its behavior on other imaging equipment or patient
populations is untested.
