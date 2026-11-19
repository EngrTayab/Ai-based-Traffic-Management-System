AI-Based Smart Traffic Light Management System
Prototype for the PHEC Honhaar PINTEC Expo 2026.
Muhammad Tayyab Asif  · Supervisor: Dr. Sunnia Ikram
The Islamia University of Bahawalpur, Department of Software Engineering
Educational prototype only. It does not control real traffic and is
not a certified traffic safety system.
What is built in this version (v0.32)
Part
File
Status
Adaptive algorithm (PCU, priority, Low/Medium/High, green time, safety cycle)
backend/traffic_algorithm.py
Working, tested
Fixed vs adaptive simulation comparison
backend/simulator.py
Working, tested
REST API + dashboard server
backend/main.py
Working, tested
Experiment storage (SQLite, no server needed)
backend/storage.py
Working
Web dashboard (counts, levels, lights, countdown, comparison, live camera)
frontend/index.html
Working, tested
Video upload analysis (one video per approach, or one ROI video)
backend/video_analysis.py
Working, tested
Live camera (one approach)
backend/video_analysis.py
Working, tested
Live-model mode (one camera watches all four roads)
backend/video_analysis.py
Working, tested
Auto-fit regions (detect the four road regions from the camera frame)
backend/video_analysis.py
Working, tested
Emergency + pedestrian override buttons
backend/main.py
Working, tested
Manual light test (set any light by hand)
backend/main.py
Working, tested
Printable report
backend/main.py
Working
Vehicle tracking (opt-in, ByteTrack)
backend/video_analysis.py
Working
Standalone detection test (webcam or video file)
backend/detection.py
Working
Tests
tests/
11 tests, all passing
Not in this version: login (decided not needed for the local demo),
MongoDB (SQLite is used instead — lighter, no server to install),
ESP32 hardware kit (optional, not included in this repo).
How it works, in simple words
Vehicle counts come in for North, East, South and West — from YOLO
   detection on a video or live camera, or from manual numbers.
Each approach gets a density score using PCU weights
   (motorcycle 0.5, car 1.0, truck 2.0, bus 2.5) and a priority score:
   priority = 0.6 * density + 0.4 * queue.
Density becomes a level: Low (< 5), Medium (5–15), High (> 15).
Green time is shared in proportion to priority, always between
   10 and 60 seconds, followed by 3 s yellow and 2 s all-red.
   Only one approach is ever green at a time. The busiest approach
   goes first.
The dashboard shows all of this live, and can run a simulation
   comparing fixed-time (30 s each) with adaptive timing on the
   same traffic scenario.
Run it — Windows, PowerShell
Step 0: Install Python
Download Python 3.12 or 3.13 from https://www.python.org/downloads/
and install it. During the install, tick the box
"Add Python to PATH".
Step 1: Install Microsoft Visual C++ Redistributable (Windows only)
PyTorch (the AI library) needs this on Windows. Without it, the
server fails to start with an error like:
OSError: [WinError 126] The specified module could not be found
Download and install it (free, one click, no restart needed):
https://aka.ms/vs/17/release/vc_redist.x64.exe
If you already have it installed, skip this step.
Step 2: Get the code
git clone https://github.com/EngrTayab/Ai-based-Traffic-Management-System.git
cd Ai-based-Traffic-Management-System/smart-traffic-system
Or download the ZIP from the GitHub page (Code → Download ZIP)
and extract it.
Step 3: Create a virtual environment and install the libraries
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
If activation is blocked, run this once and try again:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
This installs FastAPI, Uvicorn, Ultralytics (YOLO), OpenCV and
PyTorch. PyTorch is large (about 200 MB), so the download takes
a few minutes on a slow connection.
Light install (API only, no detection):
pip install -r backend/requirements-core.txt.
The dashboard and simulation work, but video upload and the live
camera need the full install above.
Step 4: Run the tests (optional)
python -m unittest discover -s tests -v
Expected: all tests pass (OK at the end).
Step 5: Start the backend and open the dashboard
cd backend
uvicorn main:app --port 8000
Open http://127.0.0.1:8000 for the dashboard and
http://127.0.0.1:8000/docs for the API documentation.
Stop the server with Ctrl + C.
On startup the dashboard shows "No measurement yet" — the
signals stay stopped (all red) until the first real measurement
arrives. No demo numbers are shown.
Step 6: Upload videos from the dashboard
Choose a video for each approach you have (at least one video is
required), and click Analyze videos & recommend signal times.
The system samples up to 20 frames per video, counts the vehicles
with YOLO, averages them, and recommends a green time for every
signal. The first analysis uses yolov8n.pt (included in the repo,
about 6 MB) and takes about 20–30 seconds per video on a normal
laptop CPU.
You can also upload a single video of a whole intersection
(ROI mode) — draw the four regions on the dashboard and the system
counts each approach from the one video.
You can also test detection on its own, without the dashboard:
python backend/detection.py --source 0
python backend/detection.py --source data/videos/test.mp4
Press q to quit.
API endpoints
Method
Endpoint
Purpose
GET
/api/status
Counts, levels, green times, current phase
POST
/api/analyze-videos
Upload one video per approach, get counts, levels and recommended green times
POST
/api/analyze-intersection
Upload one video of a whole intersection with ROI regions, get per-approach counts
POST
/api/mode
Switch adaptive / fixed mode
POST
/api/evaluate
Send counts for 4 approaches, get green times
GET
/api/config
Current algorithm configuration
GET
/api/scenarios
Low / medium / high simulation scenarios
POST
/api/experiments/run?scenario=medium
Run fixed vs adaptive simulation and store it
GET
/api/experiments
Stored experiment runs
POST
/api/emergency
Emergency green for one approach (20 s)
POST
/api/pedestrian
Pedestrian crossing — all red for 12 s
POST
/api/override/cancel
Cancel the current emergency/pedestrian override
POST
/api/manual-light
Set one light by hand (manual test)
POST
/api/manual-light/cancel
Back to automatic signals
POST
/api/live-frame
Send one live-camera frame for one approach
POST
/api/live-stop
Stop the one-road live camera
POST
/api/live-model-frame
Send one live frame of the whole model (all four roads)
POST
/api/live-model-stop
Stop the live-model camera
POST
/api/auto-regions
Detect the four road regions from a camera frame
GET
/report
Printable project report page
About the comparison numbers
simulator.py compares both strategies on the **same assumed arrival
rates**. Its waiting-time, queue and throughput numbers are simulation
estimates, not measurements from real traffic, and not a promise that
adaptive always wins. During the evaluation stage, the arrival rates
should be calibrated from the real test videos, and detection accuracy
should be reported against manual counts.
About the video counting method
For each uploaded video the system averages the vehicles **visible
per sampled frame**. That is a density estimate ("how crowded is
this road on average"), not a count of unique vehicles, and the
queue length is approximated by the same average. Separating stopped
vehicles from moving ones, and tracking unique vehicles, are later
improvements that need object tracking. Detection accuracy should
be checked against manual counts during the evaluation stage and
reported as measured.
Project structure
smart-traffic-system/
  README.md
  backend/
    config.json            All tunable values (PCU, thresholds, timings)
    traffic_algorithm.py   The adaptive signal logic (no heavy libraries)
    simulator.py           Fixed vs adaptive simulation
    video_analysis.py      Video + live-camera analysis (frame sampling + YOLO counting)
    detection.py           YOLO vehicle detection (standalone test)
    storage.py             SQLite experiment storage
    main.py                FastAPI backend + dashboard server
    requirements-core.txt  Light install: API only
    requirements.txt       Full install: API + detection
    yolov8n.pt             YOLO nano model (included, ~6 MB)
  frontend/
    index.html             Dashboard (no build step needed)
  tests/
    test_algorithm.py      Algorithm + simulation tests
    test_video_analysis.py Video analysis tests
  data/
    videos/                Put test videos here (not included)
