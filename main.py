import os
import shutil

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, FileResponse

from anonymizer import anonymize_file

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/upload/")
async def upload_file(file: UploadFile):
    file_destination = f"./user_temp_files/{file.filename}"
    with open(file_destination, "wb") as file_path:
        shutil.copyfileobj(file.file, file_path)
    return JSONResponse(content={"message": "Files uploaded successfully!", "file_name": file.filename})

@app.get("/anonymize/{file_name}")
async def anonymize(file_name: str):
    file_name = anonymize_file(file_name)
    return JSONResponse(content={"message": "File successfully anonymized and saved!", "file_name": file_name})

@app.get("/download/{file_name}")
async def download_file(file_name: str):
    path_to_file = os.path.join("user_temp_files", file_name)
    return FileResponse(path=path_to_file)
