const videoElement = document.getElementById('inputVideo');
const canvasElement = document.getElementById('outputCanvas');
const canvasCtx = canvasElement.getContext('2d');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const alertLog = document.getElementById('alertLog');
const snapshots = document.getElementById('snapshots');

let lastFaceCenter = null;
let cheatingCounter = 0;
let stableCounter = 0;
let isCurrentlySuspicious = false;
let lastSnapshotTime = 0;
let phoneModel;
let examActive = false;
let timerInterval;
let remainingTime = 600; // 10 minutes in secon

const CHEATING_THRESHOLD = 5;
const STABLE_THRESHOLD = 5;
const SNAPSHOT_COOLDOWN = 5;

function getFaceCenter(landmarks) {
  let sumX = 0, sumY = 0;
  for (let point of landmarks) {
    sumX += point.x;
    sumY += point.y;
  }
  return { x: sumX / landmarks.length, y: sumY / landmarks.length };
}

function isFaceMoving(current, previous) {
  if (!previous) return false;
  const dx = current.x - previous.x;
  const dy = current.y - previous.y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  return distance > 0.01;
}

function takeSnapshot() {
  const now = Date.now();
  if (now - lastSnapshotTime < SNAPSHOT_COOLDOWN || !examActive) return;

  const img = document.createElement('img');
  img.className = 'snapshot';
  img.src = canvasElement.toDataURL('image/jpeg');
  
  // Add to beginning of container
  if (snapshots.firstChild) {
    snapshots.insertBefore(img, snapshots.firstChild);
  } else {
    snapshots.appendChild(img);
  }
  
  // Limit to 6 snapshots
  if (snapshots.children.length > 1) {
    snapshots.removeChild(snapshots.lastChild);
  }
  
  lastSnapshotTime = now;
}

function addAlert(type, message) {
  const alertItem = document.createElement('div');
  alertItem.className = 'alert-item';
  
  const icon = document.createElement('div');
  icon.className = `alert-icon ${type}`;
  icon.innerHTML = type === 'danger' ? '<span class="material-icons">error</span>' : '<span class="material-icons">warning</span>';
  
  const content = document.createElement('div');
  const time = new Date().toLocaleTimeString();
  content.innerHTML = `<div>${message}</div><div class="alert-time">${time}</div>`;
  
  alertItem.appendChild(icon);
  alertItem.appendChild(content);
  
  // Add to top of log
  if (alertLog.firstChild) {
    alertLog.insertBefore(alertItem, alertLog.firstChild);
  } else {
    alertLog.appendChild(alertItem);
  }
}

function updateStatus(state, message) {
  statusText.textContent = message;
  statusDot.className = 'status-dot';
  
  switch(state) {
    case 'ready':
      statusDot.classList.add('active');
      break;
    case 'warning':
      statusDot.style.backgroundColor = 'var(--warning)';
      break;
    case 'danger':
      statusDot.style.backgroundColor = 'var(--danger)';
      break;
    case 'inactive':
      statusDot.style.backgroundColor = 'var(--gray)';
      break;
  }
}

async function loadPhoneModel() {
  phoneModel = await cocoSsd.load();
  console.log("Phone detection model loaded");
  updateStatus('ready', 'System ready. Click "Start Exam" to begin.');
}

async function detectPhoneFromVideoFrame() {
  if (!phoneModel || !examActive) return;

  tf.engine().startScope();
  const tensor = tf.browser.fromPixels(videoElement);
  const predictions = await phoneModel.detect(tensor);
  tf.engine().endScope();

  const phoneDetected = predictions.some(pred => 
    pred.class === 'cell phone' && pred.score > 0.6
  );
  
  if (phoneDetected) {
    cheatingCounter++;
    stableCounter = 0;
    if (cheatingCounter >= CHEATING_THRESHOLD && !isCurrentlySuspicious) {
      isCurrentlySuspicious = true;
      updateStatus('danger', 'Phone detected!');
      addAlert('danger', 'Mobile device detected');
      takeSnapshot();
    }
  }
}

const faceMesh = new FaceMesh({
  locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});

faceMesh.setOptions({
  maxNumFaces: 3,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

faceMesh.onResults(results => {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (!examActive) {
    canvasCtx.restore();
    return;
  }

  const faces = results.multiFaceLandmarks || [];

  // Multiple faces detected
  if (faces.length > 1) {
    cheatingCounter++;
    stableCounter = 0;
    if (cheatingCounter >= CHEATING_THRESHOLD && !isCurrentlySuspicious) {
      isCurrentlySuspicious = true;
      updateStatus('danger', `Multiple faces detected (${faces.length})`);
      addAlert('danger', `Multiple faces detected (${faces.length})`);
      takeSnapshot();
    }
    canvasCtx.restore();
    return;
  }

  // No face detected
  if (faces.length === 0) {
    cheatingCounter++;
    stableCounter = 0;
    if (cheatingCounter >= CHEATING_THRESHOLD && !isCurrentlySuspicious) {
      isCurrentlySuspicious = true;
      updateStatus('danger', 'Candidate not in frame');
      addAlert('danger', 'Face not detected');
      takeSnapshot();
    }
    canvasCtx.restore();
    return;
  }

  // Single face - check movement
  const landmarks = faces[0];
  const faceCenter = getFaceCenter(landmarks);
  const moving = isFaceMoving(faceCenter, lastFaceCenter);
  lastFaceCenter = faceCenter;

  if (moving) {
    cheatingCounter++;
    stableCounter = 0;
    if (cheatingCounter >= CHEATING_THRESHOLD && !isCurrentlySuspicious) {
      isCurrentlySuspicious = true;
      updateStatus('warning', 'Suspicious head movement');
      addAlert('warning', 'Excessive head movement');
      takeSnapshot();
    }
  } else {
    stableCounter++;
    if (stableCounter >= STABLE_THRESHOLD) {
      cheatingCounter = 0;
      if (isCurrentlySuspicious) {
        isCurrentlySuspicious = false;
        updateStatus('ready', 'Candidate stable');
      }
    }
  }

  canvasCtx.restore();
});

const camera = new Camera(videoElement, {
  onFrame: async () => {
    await faceMesh.send({ image: videoElement });
    detectPhoneFromVideoFrame();
  },
  width: 640,
  height: 480
});

function startTimer() {
  // Clear any existing timer
  clearInterval(timerInterval);
  
  // Update the timer display immediately
  updateTimer(remainingTime);
  
  // Start new interval
  timerInterval = setInterval(() => {
    remainingTime--;
    updateTimer(remainingTime);
    
    if (remainingTime <= 0) {
      clearInterval(timerInterval);
      endTest();
    }
  }, 1000);
}

function resetTimer() {
  clearInterval(timerInterval);
  remainingTime = 600; // Reset to 10 minutes
  updateTimer(remainingTime);
}

function updateTimer(seconds) {
  const hours = String(Math.floor(seconds / 3600)).padStart(2, '0');
  const minutes = String(Math.floor((seconds % 3600) / 60)).padStart(2, '0');
  const secs = String(seconds % 60).padStart(2, '0');
  document.getElementById("timer").textContent = `${hours}:${minutes}:${secs}`;
  
  // Change color when less than 5 minutes remain
  if (seconds < 300) {
    document.getElementById("timer").style.color = 'var(--danger)';
  }
}


function startProctoring() {
  examActive = true;
  camera.start();
  updateStatus('ready', 'Exam in progress. Monitoring active.');
  startTimer();
  
  const endBtn = document.getElementById('endBtn');
  endBtn.classList.remove('disabled');
  endBtn.disabled = false;
  
  const startBtn = document.getElementById('startBtn');
  startBtn.classList.add('disabled');
  startBtn.disabled = true;
}

function stopProctoring() {
  examActive = false;
  camera.stop();
  updateStatus('inactive', 'Exam ended. No longer monitoring.');
  resetTimer();
  
  const startBtn = document.getElementById('startBtn');
  startBtn.classList.remove('disabled');
  startBtn.disabled = false;
  
  const endBtn = document.getElementById('endBtn');
  endBtn.classList.add('disabled');
  endBtn.disabled = true;
}

function closeInstructions() {
  document.getElementById('instructionModal').style.display = 'none';
}

// Initialize
loadPhoneModel();

// Export functions for the HTML buttons
window.startTest = function() {
  startProctoring();
};

window.endTest = function() {
  stopProctoring();
};