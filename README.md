# CORA OpenCLIP image-relevance verifier (Render starter)

> OpenCLIP is a similarity model, **not** a fully validated out-of-scope detector. This starter defaults to advisory `review_only` until you evaluate real images and explicitly enable automatic rejection. It does not perform image classification for CORA; MobileNetV3-Small remains the primary classifier.

## 1. Upload project to GitHub

Create a **new repository** such as `cora-openclip-verifier`. Upload `main.py`, `requirements.txt`, `build.sh`, `.python-version`, and `.gitignore`. Do not upload your API token, `.env`, or private photos.

## 2. Create a new Render Web Service

- New > Web Service > connect the new GitHub repo
- Language/runtime: Python
- Build command: `bash build.sh`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Select a plan with sufficient RAM for CPU PyTorch and OpenCLIP; very small/free instances may fail to start or time out. As a starting point, budget for roughly 2 GB or more, but verify actual memory consumption and current Render plans.
- Set environment variables:
  - `API_TOKEN`: long random secret (mandatory)
  - `AUTO_REJECT_ENABLED=false` (mandatory for first tests)
  - optionally `TORCH_NUM_THREADS=2`, `MAX_UPLOAD_MB=6`
- Deploy and check `https://YOUR-SERVICE.onrender.com/health` returns `{"status":"ok","ready":true}`.
- Render build must be able to download the pretrained model weights. If the build fails, inspect Logs; check storage/RAM/network and whether the pretrained model identifier remains supported.

## 3. Test from Postman or curl

Endpoint: `POST https://YOUR-SERVICE.onrender.com/verify`

Header: `X-API-Key: your_API_TOKEN`

Body type: `form-data`
- `file` (File): JPG, PNG or WEBP test image
- `predicted_category` (Text): one of the **exact** strings below

```
Improper Waste Disposal
Streetlight Problem
Drainage/Flooding Concern
Road Damage
```

Example curl (Windows PowerShell):

```powershell
curl.exe -X POST "https://YOUR-SERVICE.onrender.com/verify" `
  -H "X-API-Key: YOUR_API_TOKEN" `
  -F "file=@C:\Users\YOU\Pictures\test-road.jpg" `
  -F "predicted_category=Road Damage"
```

Example response (numbers are **illustrative, not tested values**):

```json
{
  "predicted_category": "Road Damage",
  "openclip_best_category": "Road Damage",
  "category_matches": true,
  "unrelated_example_competes": false,
  "predicted_category_similarity": 0.0,
  "highest_unrelated_similarity": 0.0,
  "category_similarities": {"Improper Waste Disposal": 0.0, "Streetlight Problem": 0.0, "Drainage/Flooding Concern": 0.0, "Road Damage": 0.0},
  "decision": "review_only",
  "auto_reject_enabled": false,
  "note": "Similarity scores are NOT confidence probabilities. Validate thresholds on local data."
}
```

## 4. Calibrate before auto-reject

Gather real examples from all four CORA categories plus unrelated images (selfies, food, pets, unrelated streets, posters) and unclear-but-valid cases. For each sample, record MobileNet prediction, OpenCLIP scores, and a human-verified relevance label.

- Calculate false rejection rate for valid images and false acceptance rate for unrelated ones.
- Try different category prompts if necessary, retest, and choose an evidence-based `MIN_RELEVANCE_SCORE` (cosine similarity, **not confidence**).
- Set `AUTO_REJECT_ENABLED=true` and `MIN_RELEVANCE_SCORE=<validated value>` **only after calibration**.
- When enabled, decisions become `accepted` or `rejected`, based on prompt-based category match, negative examples, and the calibrated threshold.
- Genuine ambiguous images can be falsely rejected. Keep camera recapture and a suitable review/correction path. Do not confuse API errors with a confirmed reject.

**A threshold cannot be responsibly supplied as a universal constant.**

## 5. Connect to CORA

Recommended data flow:

```
Resident camera -> CORA backend -> MobileNetV3-Small Render API
                              -> OpenCLIP Render /verify (image + MobileNet label)
                              -> handle accept/reject/review-only
                              -> after validation, optionally Gemini auto-details
```

Store these **only in the CORA backend**:

```env
OPENCLIP_API_URL=https://YOUR-SERVICE.onrender.com
OPENCLIP_API_TOKEN=the-same-secret-as-API_TOKEN
```

Sample Node.js 20+ backend function:

```js
async function verifyImageWithOpenCLIP(imageBuffer, fileName, mimeType, category) {
  const form = new FormData();
  form.append("file", new Blob([imageBuffer], { type: mimeType }), fileName);
  form.append("predicted_category", category);

  const response = await fetch(`${process.env.OPENCLIP_API_URL}/verify`, {
    method: "POST",
    headers: { "X-API-Key": process.env.OPENCLIP_API_TOKEN },
    body: form,
    signal: AbortSignal.timeout(60000),
  });
  if (!response.ok) throw new Error(`OpenCLIP verification failed: ${response.status}`);
  return await response.json();
}
```

**Do not put the `OPENCLIP_API_TOKEN` into a frontend `VITE_` variable.** Validate the user session and file on the existing backend before calling OpenCLIP. On `review_only`, do not treat the image as officially verified. Render instances may cold start; avoid repeated calls.

## Notes

- The deployment has **no third-party OpenCLIP API quota** but is limited by Render RAM, CPU, uptime, concurrency, and cost.
- A CLIP text-image similarity score is **not** a probability of relevance.
- This code is starter code. You must verify model output with real CORA images and ensure it meets your thesis validation requirements.
