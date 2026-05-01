# RefCheck AI

RefCheck AI is a Django web app for reviewing short basketball clips. It lets a user upload a play, provide the original call, and receive a structured analysis that separates visible evidence from rule-based verdict reasoning.

The project is built around a practical officiating workflow: find useful moments in the video, describe what is visible, retrieve relevant basketball rules when configured, then produce a cautious verdict on whether the original call appears fair, bad, or inconclusive.

## Core Functionality

- Upload MP4, MOV, or WEBM basketball clips up to 100 MB.
- Generate signed Google Cloud Storage upload URLs for browser-based video uploads.
- Extract representative evidence frames with OpenCV using temporal coverage, motion, sharpness, and event-phase heuristics.
- Use Gemini as a visual analyst to summarize the sequence, identify primary events, flag possible officiating issues, and recommend frames for verdict review.
- Optionally retrieve basketball rule context through OpenAI embeddings and a Pinecone index.
- Run a second Gemini verdict stage that reviews the original call using selected frames, visual findings, and available rule context.
- Display the uploaded video, analysis status, structured visual findings, verdict, confidence, limitations, and debug details in Django templates.

## How It Works

1. The user uploads a short basketball clip and enters the original call, such as "Blocking foul" or "Goaltending".
2. The frontend requests a signed upload URL from `/api/create-upload-url`, uploads the video to GCS, then asks Django to analyze the stored object.
3. The backend downloads the clip, samples candidate frames, and selects evidence frames that preserve before/during/after context.
4. Agent 1, the visual analyst, produces a neutral JSON description of the clip without making rules conclusions.
5. If available, rule retrieval gathers relevant rule snippets from Pinecone using OpenAI embeddings.
6. Agent 2, the verdict agent, reviews the original call and returns a structured decision: `Fair Call`, `Bad Call`, or `Inconclusive`.

## Tech Stack

- Django 6 for routing, server-rendered views, sessions, and upload orchestration.
- Tailwind CDN and Django templates for the current UI.
- OpenCV and NumPy for video metadata, frame extraction, and candidate scoring.
- Google Gemini for multimodal frame analysis and verdict generation.
- Google Cloud Storage for direct browser uploads and signed video reads.
- OpenAI embeddings and Pinecone for optional basketball rule retrieval.
- SQLite by default for local Django state.

## Getting Started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000`.

The app loads environment variables from `.env.local` when `python-dotenv` is installed.

## Configuration

Minimum useful local configuration:

```bash
DJANGO_SECRET_KEY=replace-me
DJANGO_DEBUG=True
GEMINI_API_KEY=your-gemini-key
```

Direct browser uploads require Google Cloud Storage:

```bash
GCP_STORAGE_BUCKET_NAME=your-bucket
GOOGLE_APPLICATION_CREDENTIALS_JSON='{"type":"service_account",...}'
REFCHECK_GCS_CORS_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

Rule retrieval is optional. When these values are absent, the app still runs the visual pipeline and verdict stage, but the verdict will not include retrieved rule snippets.

```bash
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=nba-rules
PINECONE_NAMESPACE=
```

Useful tuning values:

```bash
REFCHECK_SCAN_FPS=3
REFCHECK_DENSE_FPS=15
REFCHECK_DENSE_WINDOW_SECONDS=1.25
REFCHECK_MIN_EVIDENCE_FRAMES=12
REFCHECK_MAX_EVIDENCE_FRAMES=20
REFCHECK_MAX_VERDICT_FRAMES=8
REFCHECK_MAX_GEMINI_IMAGES=24
REFCHECK_MAX_CANDIDATE_FRAMES=80
GEMINI_VISUAL_MODEL=gemini-2.5-flash-lite
GEMINI_VERDICT_MODEL=gemini-2.5-flash-lite
GEMINI_FALLBACK_MODEL=gemini-2.5-flash
GEMINI_MAX_RETRIES=3
```

## Tests

```bash
python manage.py test
```

The test suite covers GCS upload URL validation, environment-based credentials, frame selection behavior, visual prompt contracts, rule-aware verdict prompting, and verdict safeguards for ambiguous evidence.

## Current Constraints

- Basketball is the only supported sport in the current interface and prompt design.
- Results depend heavily on clip quality, camera angle, and whether the key moment is visible.
- The verdict is advisory. It is not an official referee decision system.
- The browser upload flow is designed around GCS signed uploads; server-side local upload support exists, but the current template path expects cloud upload configuration.
- The default Django settings are development-oriented: SQLite, a fallback dev secret, synchronous analysis, and no background job queue.
- Rule retrieval depends on an existing Pinecone index populated with suitable basketball rule text.

## Project Structure

```text
core/
  services.py      Video frame selection, Gemini calls, rule retrieval, verdict pipeline
  views.py         Upload endpoints, GCS helpers, analysis orchestration
  urls.py          App routes
  tests.py         Unit tests for upload, selection, prompts, and verdict behavior
refcheck/
  settings.py      Django settings and environment loading
  urls.py          Project URL routing
templates/
  core/            Home, analysis, result, and about pages
```
