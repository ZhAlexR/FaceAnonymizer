import os
import shutil
from typing import Literal

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from anonymizer import process_and_save_anonymized_file

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload/")
async def upload_file(file: UploadFile):
    file_destination = f"./user_temp_files/{file.filename}"
    with open(file_destination, "wb") as file_path:
        shutil.copyfileobj(file.file, file_path)
    return JSONResponse(content={"message": "Files uploaded successfully!", "file_name": file.filename})

@app.get("/anonymize/{file_name}")
async def anonymize(file_name: str, file_type: Literal["photo", "video"]):
    file_name = process_and_save_anonymized_file(file_name, file_type)
    return JSONResponse(content={"message": "File successfully anonymized and saved!", "file_name": file_name})

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    path_to_file = os.path.join("user_temp_files", file_name)
    return FileResponse(
        path=path_to_file,
        filename=file_name,
        media_type='application/octet-stream'
    )
