import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import argparse
import numpy as np
from modules.commons import str2bool
from inference import load_models, process_audio
from inference import converter
app = FastAPI()


@app.get("/")
async def read_index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            input_chunk = await websocket.receive_bytes()
            print("Received chunk of size", len(input_chunk))
            input_chunk = np.frombuffer(input_chunk, dtype=np.float32)
            output_chunk = converter.stream_input(input_chunk)
            if output_chunk is not None:
                await websocket.send_bytes(output_chunk.tobytes())

    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, log_level="info")