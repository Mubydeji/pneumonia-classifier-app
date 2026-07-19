# Chest X-Ray Pneumonia Classifier — Web App

A deployed inference service for the [pneumonia classifier](https://github.com/Mubydeji/pneumonia-classifier)
trained in a companion notebook. FastAPI backend, static HTML/CSS/JS frontend,
containerized and deployed to Google Cloud Run. Model weights are hosted on
Hugging Face Hub and downloaded at container startup, rather than bundled
into the Docker image or committed to this repo.

**Live demo:** [pneumonia-classifier-app.onrender.com](https://pneumonia-classifier-app.onrender.com)

## Architecture

```
Browser (static frontend)
        │  POST /predict (image file)
        ▼
FastAPI app (Render, Docker web service)
        │  downloads weights from Hugging Face Hub at startup
        ▼
ResNet-50 (CPU inference)
        │
        ▼
{ probability, threshold, label }
```

**Note on the free tier:** this is deployed on Render's free web service tier, which
spins down after 15 minutes of inactivity. The first request after idle time will be
slow (30-60 second cold start) while the container restarts and re-downloads the
214MB model checkpoint from Hugging Face. Subsequent requests are fast.

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

## Deploying to Render

This app is deployed on [Render](https://render.com)'s free Docker web service tier.

1. Push this repo to GitHub.
2. On Render: **New +** → **Web Service** → connect this repo.
3. Render auto-detects the `Dockerfile`. Select the **Free** instance type.
4. Add environment variables:
   - `MODEL_REPO_ID` = `muby26/chest-xray-pneumonia-resnet50`
   - `MODEL_FILENAME` = `resnet50_pneumonia.pt`
5. Click **Create Web Service**. Render builds the Docker image and deploys automatically
   on every push to `main`.

The same Dockerfile also works unmodified on any other container platform that accepts
a standard Docker image and passes a `$PORT` environment variable (e.g. Google Cloud
Run, Fly.io), if a different host is preferred later.

## Limitations

This is a research prototype, not a validated clinical tool. See the
[training repository](https://github.com/Mubydeji/pneumonia-classifier)
for the full methodology, evaluation, and stated limitations, including
the fact that the model was trained and tested on a single-institution
dataset and its behavior on other imaging equipment or patient
populations is untested.
