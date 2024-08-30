from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image
import io

app = FastAPI

# Load Yolov10 model # (실제 우리가 만든 YOLOv10을 로딩하는 방법)
# model = load_yolov10_model('yolov10.pth') # 확장자는 조금 달라질 수 있음. 
# model.eval()

# Load GPT-2 model # 예시
tokenizer =  GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

class TextRequest(BaseModel):
    text: str

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    #predictions = yolo_predict(image)
    return{"result": "dummy_image_result"} # Replace with actual result

@app.post("/generate_text/")
async def generate_text(request: TextRequest):
    inputs = tokenizer.encode(request.text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": generated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)