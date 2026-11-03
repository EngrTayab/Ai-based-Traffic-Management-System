# AI-Based Smart Traffic Light Management System

Prototype for the PHEC Honhaar PINTEC Expo 2026.
Muhammad Tayyab Asif · Co-author: Ahmad Ramzan · Supervisor: Dr. Sunnia Ikram
The Islamia University of Bahawalpur, Department of Software Engineering

> Educational prototype only. It does not control real traffic and is
> not a certified traffic safety system.

## What is built in this version (v0.2)

| Part | File | Status |
| --- | --- | --- |
| Adaptive algorithm (PCU, priority, Low/Medium/High, green time, safety cycle) | `backend/traffic_algorithm.py` | Working, tested |
| Fixed vs adaptive simulation comparison | `backend/simulator.py` | Working, tested |
| REST API + status / evaluate / experiments / video upload | `backend/main.py` | Working, tested |
| Experiment storage (SQLite, no server needed) | `backend/storage.py` | Working |
| Web dashboard (upload videos, counts, levels, lights, countdown, comparison) | `frontend/index.html` | Working, tested |
| Vehicle detection (YOLOv8n, car/motorcycle/bus/truck) | `backend/detection.py` | Code ready — needs your video/webcam test on your laptop |
| Video analysis (sample frames from an uploaded video, average counts per approach) | `backend/video_analysis.py` | Pipeline tested; real YOLO counting needs your laptop test |
| Tests | `tests/` | 11 tests, all passing |

### New in v0.2: upload a traffic video, get signal times

The dashboard now has an **Upload traffic videos** section. One normal
video shows one road, so there is one upload per approach
(North / East / South / West). The system samples frames from each
video (at most 20 per video), counts the vehicles visible in those
frames with YOLO, averages them, and uses that density to recommend
a green time for every signal.

Until a video is uploaded, the dashboard numbers are clearly labelled
**demo** numbers — they are sample numbers built into the code so the
dashboard can be tested, not measurements of any real traffic.

Not in this version: login (decided not needed for the local demo),
MongoDB (SQLite is used instead — lighter, no server to install),
Arduino/ESP32 hardware and emergency-vehicle detection (optional,
from the SRS, only after the software core is demonstrated).

## How it works, in simple words

1. Vehicle counts come in for North, East, South and West — from YOLO
   detection on a video, or from sample/manual numbers.
2. Each approach gets a density score using PCU weights
   (motorcycle 0.5, car 1.0, truck 2.0, bus 2.5) and a priority score:
   `priority = 0.6 * density + 0.4 * queue`.
3. Density becomes a level: Low (< 5), Medium (5–15), High (> 15).
4. Green time is shared in proportion to priority, always between
   10 and 60 seconds, followed by 3 s yellow and 2 s all-red.
   Only one approach is ever green at a time.
5. The dashboard shows all of this live, and can run a simulation
   comparing fixed-time (30 s each) with adaptive timing on the
   same traffic scenario.

## Run it — Windows, PowerShell

### Step 1: Install the light core (fast, recommended first)

```powershell
cd smart-traffic-system
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend/requirements-core.txt
```

If activation is blocked, run this once and try again:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### Step 2: Run the tests

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass (`OK` at the end).

### Step 3: Start the backend and open the dashboard

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

Open http://127.0.0.1:8000 for the dashboard and
http://127.0.0.1:8000/docs for the API documentation.
Stop the server with `Ctrl + C`.

On startup the dashboard shows clearly labelled **demo** numbers,
because no video is connected yet. Those are sample numbers, not
measurements.

### Step 4: Upload videos from the dashboard

Video analysis needs the detection libraries, which are large
(PyTorch), so they are a separate install:

```powershell
pip install -r backend/requirements.txt
```

Restart the backend, refresh the dashboard, choose a video for each
approach you have, and click
**Analyze videos & recommend signal times**.
The first analysis downloads `yolov8n.pt` (about 6 MB) and can take
a while on CPU — the system checks 20 frames per video, not every
frame. The result table shows, for each signal, its traffic level
and recommended green seconds, sorted from busiest to quietest.

You can also test detection on its own, without the dashboard:

```powershell
python backend/detection.py --source 0
python backend/detection.py --source data/videos/test.mp4
```

Press `q` to quit.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/api/status` | Counts, levels, green times, current phase |
| POST | `/api/analyze-videos` | Upload one video per approach (form fields `north`, `east`, `south`, `west`), get counts, levels and recommended green times |
| POST | `/api/mode` | Switch adaptive / fixed mode only |
| POST | `/api/evaluate` | Send counts for 4 approaches, get green times |
| GET | `/api/config` | Current algorithm configuration |
| GET | `/api/scenarios` | Low / medium / high simulation scenarios |
| POST | `/api/experiments/run?scenario=medium` | Run fixed vs adaptive simulation and store it |
| GET | `/api/experiments` | Stored experiment runs |

## About the comparison numbers

`simulator.py` compares both strategies on the **same assumed arrival
rates**. Its waiting-time, queue and throughput numbers are simulation
estimates, not measurements from real traffic, and not a promise that
adaptive always wins. During the evaluation stage, the arrival rates
should be calibrated from the real test videos, and detection accuracy
should be reported against manual counts.

## About the video counting method

For each uploaded video the system averages the vehicles **visible
per sampled frame**. That is a density estimate ("how crowded is
this road on average"), not a count of unique vehicles, and the
queue length is approximated by the same average. Separating stopped
vehicles from moving ones, and tracking unique vehicles, are later
improvements that need object tracking. Detection accuracy should
be checked against manual counts during the evaluation stage and
reported as measured.

## Project structure

```
smart-traffic-system/
  README.md
  backend/
    config.json            All tunable values (PCU, thresholds, timings)
    traffic_algorithm.py   The adaptive signal logic (no heavy libraries)
    simulator.py           Fixed vs adaptive simulation
    video_analysis.py      Uploaded-video analysis (frame sampling + counting)
    detection.py           YOLOv8n vehicle detection (standalone test)
    storage.py             SQLite experiment storage
    main.py                FastAPI backend + dashboard server
    requirements-core.txt  Light install: API only
    requirements.txt       Full install: API + detection
  frontend/
    index.html             Dashboard (no build step needed)
  tests/
    test_algorithm.py      Algorithm + simulation tests
  data/
    videos/                Put test videos here (not included)
```
