import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()

PORT = os.getenv("PORT", 8000)

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=int(PORT), reload=True)
