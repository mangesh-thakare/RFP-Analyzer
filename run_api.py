import os
import sys
import uvicorn

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.rfp_analyzer import app

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.104.79", port=8001,timeout_keep_alive=1000) 