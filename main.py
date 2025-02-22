import shutil

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse

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
    return JSONResponse(content={"message": "Files uploaded successfully!", "file_paths": file_path})