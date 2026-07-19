const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const uploadStage = document.getElementById('uploadStage');
const analysisStage = document.getElementById('analysisStage');
const previewImage = document.getElementById('previewImage');
const scanLine = document.getElementById('scanLine');
const resultPending = document.getElementById('resultPending');
const resultComplete = document.getElementById('resultComplete');
const resultError = document.getElementById('resultError');
const resultLabel = document.getElementById('resultLabel');
const resultValue = document.getElementById('resultValue');
const scoreMarker = document.getElementById('scoreMarker');
const resultNote = document.getElementById('resultNote');
const errorDetail = document.getElementById('errorDetail');
const resetButton = document.getElementById('resetButton');

const THRESHOLD = 0.30;

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});

['dragover', 'dragenter'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  })
);

['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  })
);

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

resetButton.addEventListener('click', () => {
  analysisStage.hidden = true;
  uploadStage.hidden = false;
  fileInput.value = '';
});

function handleFile(file) {
  if (!file.type.match(/image\/(jpeg|png)/)) {
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImage.src = e.target.result;
  };
  reader.readAsDataURL(file);

  uploadStage.hidden = true;
  analysisStage.hidden = false;

  resultPending.hidden = false;
  resultComplete.hidden = true;
  resultError.hidden = true;
  scanLine.classList.add('scanning');

  runInference(file);
}

async function runInference(file) {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${response.status})`);
    }

    const data = await response.json();
    showResult(data);
  } catch (err) {
    showError(err.message);
  }
}

function showResult(data) {
  scanLine.classList.remove('scanning');
  resultPending.hidden = true;
  resultComplete.hidden = false;

  const probability = data.probability;
  const isFlagged = probability >= THRESHOLD;

  resultLabel.textContent = isFlagged ? 'Pneumonia indicated' : 'No pneumonia indicated';
  resultLabel.className = 'result-label ' + (isFlagged ? 'is-flagged' : 'is-normal');
  resultValue.textContent = `p = ${probability.toFixed(3)}`;

  scoreMarker.style.left = `${Math.min(Math.max(probability * 100, 0), 100)}%`;

  resultNote.textContent = isFlagged
    ? 'Score is at or above the 0.30 operating threshold, tuned for sensitivity to reduce missed cases.'
    : 'Score is below the 0.30 operating threshold.';
}

function showError(message) {
  scanLine.classList.remove('scanning');
  resultPending.hidden = true;
  resultError.hidden = false;
  errorDetail.textContent = message || 'The image may be corrupted or an unsupported format.';
}
