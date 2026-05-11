# RecruitBD-AI

RecruitBD-AI is an AI-powered CV parsing and job matching platform. The application uses a FastApi backend with local Ollama integration for extracting structured data from CVs (PDF/DOCX) and matching them against a pre-indexed job database using SentenceTransformers. The frontend is built with React, Vite, Tailwind CSS, and shadcn/ui for a seamless user experience.

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- **Python 3.13+**
- **Node.js 20+**
- **pnpm** (for frontend package management)
- **uv** (for Python package management)
- **Ollama** (Running locally)

## Backend Setup

The backend handles CV parsing and job matching. It is powered by FastAPI and requires the `gemma3` model to be pulled in your local Ollama instance.

1. **Start Ollama and Pull the Model**
   Ensure Ollama is running in the background, then pull the required model:

   ```bash
   ollama pull gemma3
   ```

2. **Navigate to the Backend Directory**

   ```bash
   cd backend
   ```

3. **Install Dependencies**
   The project uses `uv` for lightning-fast dependency management.

   ```bash
   uv sync
   ```

4. **Run the Backend Server**
   Start the FastAPI development server. It will automatically download the required `all-MiniLM-L6-v2` SentenceTransformer model on the first run.

   ```bash
   # Make sure you are in the backend directory
   uv run uvicorn api.app:app --reload --port 8080
   ```

   _The backend will now be accessible at `http://localhost:8080`._

## Frontend Setup

The frontend is a modern React application.

1. **Navigate to the Frontend Directory**

   ```bash
   cd frontend
   ```

2. **Install Dependencies**

   ```bash
   pnpm install
   ```

3. **Configure Environment Variables**
   Ensure the frontend knows how to communicate with the backend. Check the `.env` file in the `frontend/` directory (or create one):

   ```env
   VITE_API_URL="http://localhost:8080"
   ```

   _(Note: The backend defaults to 8080, so ensure this matches)._

4. **Run the Frontend Server**

   ```bash
   pnpm dev
   ```

   _The frontend will typically run at `http://localhost:5173`._

## Usage

1. Open your browser and navigate to the frontend URL (e.g., `http://localhost:5173`).
2. Upload a PDF or DOCX file of a CV.
3. The backend will parse the CV using the local `gemma3` model and immediately suggest top job matches from its indexed database.

## Architecture

- **CV Parsing**: PDF/DOCX text extraction using `pdfplumber` and `python-docx`, processed by Ollama (`gemma3`).
- **Job Matching**: Semantic similarity search powered by `sentence-transformers` (`all-MiniLM-L6-v2`) and `numpy` indexing.
- **API Engine**: FastAPI + Uvicorn.
- **Frontend**: Vite + React + Tailwind CSS + shadcn/ui.

