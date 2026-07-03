const { PoseLandmarker, FilesetResolver } = window;

const video = document.getElementById('video');
const canvas = document.getElementById('output_canvas');
const ctx = canvas.getContext('2d');
const jumpCountEl = document.getElementById('jump-count');
const startBtn = document.getElementById('start-btn');
const statusText = document.getElementById('status');
const uiPanel = document.querySelector('.ui-panel');

let poseLandmarker;
let runningMode = "VIDEO";
let lastVideoTime = -1;
let cameraActive = false;

// Jump Tracking State
let jumpCount = 0;
let isJumping = false;
let jumpFrames = 0;
let baselineY = null;

const POSE_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10], 
    [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], 
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20], [11, 23], 
    [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], 
    [28, 30], [29, 31], [30, 32], [27, 31], [28, 32]
];

// Initialize MediaPipe Tasks
async function initializeModel() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
                delegate: "CPU"
            },
            runningMode: runningMode,
            minPoseDetectionConfidence: 0.5,
            minPosePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
        });
        statusText.innerText = "Ready!";
        startBtn.style.display = "block";
    } catch (error) {
        console.error(error);
        statusText.innerText = "Error loading model. Check console.";
    }
}

initializeModel();

startBtn.addEventListener('click', async () => {
    startBtn.style.display = "none";
    statusText.style.display = "block";
    statusText.innerText = "Requesting camera access...";
    try {
        // Request camera with facingMode: user (selfie camera)
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } } 
        });
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
        cameraActive = true;
        statusText.style.display = "none";
    } catch (err) {
        console.error(err);
        statusText.innerText = "Camera access denied or unavailable.";
        startBtn.style.display = "block";
    }
});

function resizeCanvas() {
    // Keep canvas dimensions synced with the video's actual resolution
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
    }
}

function predictWebcam() {
    if (!cameraActive) return;
    resizeCanvas();
    
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        poseLandmarker.detectForVideo(video, startTimeMs, (result) => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            if (result.landmarks && result.landmarks.length > 0) {
                const landmarks = result.landmarks[0];
                drawWireframe(landmarks, canvas.width, canvas.height);
                processJumpLogic(landmarks);
            }
        });
    }
    
    // Call this function again to keep predicting when the browser is ready
    window.requestAnimationFrame(predictWebcam);
}

function drawWireframe(landmarks, w, h) {
    ctx.lineWidth = 5;
    ctx.strokeStyle = "#ff5722"; // Neon Orange
    ctx.fillStyle = "#ff5722";

    // 1. Draw Bounding Box (Green)
    const xCoords = landmarks.map(lm => lm.x * w);
    const yCoords = landmarks.map(lm => lm.y * h);
    const xMin = Math.max(0, Math.min(...xCoords));
    const xMax = Math.min(w, Math.max(...xCoords));
    const yMin = Math.max(0, Math.min(...yCoords));
    const yMax = Math.min(h, Math.max(...yCoords));
    
    ctx.strokeStyle = "#00ff00";
    ctx.lineWidth = 3;
    ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);
    
    ctx.strokeStyle = "#ff5722";

    // 2. Draw connections
    for (const [startIdx, endIdx] of POSE_CONNECTIONS) {
        const start = landmarks[startIdx];
        const end = landmarks[endIdx];
        if (start.visibility > 0.5 && end.visibility > 0.5) {
            ctx.beginPath();
            ctx.moveTo(start.x * w, start.y * h);
            ctx.lineTo(end.x * w, end.y * h);
            ctx.stroke();
        }
    }

    // 3. Draw joints
    for (const lm of landmarks) {
        if (lm.visibility > 0.5) {
            ctx.beginPath();
            ctx.arc(lm.x * w, lm.y * h, 6, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
}

function processJumpLogic(landmarks) {
    const leftHip = landmarks[23].y;
    const rightHip = landmarks[24].y;
    const avgHipY = (leftHip + rightHip) / 2;

    const leftShoulder = landmarks[11].y;
    const rightShoulder = landmarks[12].y;
    const avgShoulderY = (leftShoulder + rightShoulder) / 2;

    const torsoHeight = Math.max(0.05, avgHipY - avgShoulderY);

    if (baselineY === null) {
        baselineY = avgHipY;
    }

    const jumpThreshold = torsoHeight * 0.20;

    if (avgHipY < (baselineY - jumpThreshold)) {
        if (!isJumping) {
            isJumping = true;
            jumpFrames = 0;
        } else {
            jumpFrames++;
            // Cancel fake jumps
            if (jumpFrames > 20) {
                isJumping = false;
                baselineY = avgHipY;
            }
        }
    } else {
        if (isJumping) {
            if (avgHipY > (baselineY - jumpThreshold * 0.5)) {
                if (jumpFrames >= 2) {
                    jumpCount++;
                    updateUI();
                }
                isJumping = false;
            }
        } else {
            if (avgHipY > baselineY) {
                baselineY = baselineY * 0.5 + avgHipY * 0.5;
            } else {
                baselineY = baselineY * 0.95 + avgHipY * 0.05;
            }
        }
    }
}

function updateUI() {
    jumpCountEl.innerText = jumpCount;
    uiPanel.classList.add('bump');
    setTimeout(() => {
        uiPanel.classList.remove('bump');
    }, 150);
}
