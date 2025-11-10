# Just Decal Optimizer API

FastAPI backend for the Just Decal Optimizer front-end.

## Local run
```
pip install -r requirements.txt
uvicorn optimizer_api:app --reload
```

Then open http://127.0.0.1:8000/docs for API testing.

## Deploy on Render
1. Create a new Web Service.
2. Connect your GitHub repo.
3. Use:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn optimizer_api:app --host 0.0.0.0 --port $PORT`
4. Once deployed, you’ll get a URL like:
   ```
   https://decal-optimizer-api.onrender.com
   ```
5. Replace the API URL in your HTML:
   ```js
   fetch("https://decal-optimizer-api.onrender.com/optimize", { ... })
   ```
