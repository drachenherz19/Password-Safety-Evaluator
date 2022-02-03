
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, Request,Form
from fastapi.responses import HTMLResponse
from schema import User
import uvicorn
from fastapi.templating import Jinja2Templates
import dill
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

model=dill.load(open("xgb_classifier.pkl","rb"))
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse("index.html",context={"request":request})

vectorizer=dill.load(open("vectorizer.pkl","rb"))
async def  make_chars(inputs):
    characters=[]
    for letter in inputs:
        characters.append(letter)
    return characters


vectorizer=TfidfVectorizer(tokenizer=make_chars)
vectorizer.fit
@app.post("/",response_class=HTMLResponse)
async def predict(request:Request, password:str=Form(...)):
    x_pred=np.array([password])
    x_pred=vectorizer.transform(x_pred)
    predicted=model.predict(x_pred)
    probab=model.predict_proba(x_pred)
    if(predicted==1):
        return templates.TemplateResponse("index.html", context={"request": request, "strength": "This is an average password, it could be stronger."})
    
    if(predicted==0):
       return templates.TemplateResponse("index.html",context={"request":request,"strength":"This is an extremely weak password, prone to security threats. Consider changing it immediately."})
    if(predicted==2):
        if probab[0][2]>0.85:
            return templates.TemplateResponse("index.html", context={"request": request, "strength": "This is a rock solid password, well done!"})
        else:
            return templates.TemplateResponse("index.html", context={"request": request, "strength": "This is a strong password."})

if __name__=="__main__":
    uvicorn.run(app)
