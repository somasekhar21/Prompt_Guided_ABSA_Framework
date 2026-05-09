from fastapi import FastAPI
import uvicorn
from graph import workflow

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to the ABSA API"}
    
@app.post("/predict")
def predict(sentence:str):
    try:
        r = workflow.invoke({"sentence": sentence})
        s=f"{sentence}\n "
        for aspectSentiment in r["aspect_sentiments"].aspectSentiments:
            s+=f"{aspectSentiment.target}: {aspectSentiment.category}: {aspectSentiment.sentiment}\n"
        s+=f"\n"
        for aspectKeyword in r["aspect_keywords_mapping"].aspectKeywords.items():
            s+=f"{aspectKeyword[0]}: {aspectKeyword[1]}\n"
        s+=f"\n"
        for keyword in r["keywords"].keywords:
            s+=f"{keyword}\n"
        s+=f"\n"
        #for unknown in r["unknownList"]:
            #s+=f"{unknown}\n"
        return s 

    except Exception as e:
        print(e)
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
