from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/recommendations")
def give_food_recommendation():
    return {"recommendation": "Try Thai food tonight"}