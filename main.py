from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
import joblib
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# Configure the frontend origins allowed to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep the feature names consistent with the training pipeline.
COLUMNS = ["latitude", "longitude", "price", "minimum_nights",
    "number_of_reviews", "reviews_per_month",
    "calculated_host_listings_count", "availability_365",
    "neighbourhood_group", "neighbourhood",]

# Load the pre-trained model pipeline
model = joblib.load("Model_Pipeline.pkl")  

#Pydantic Model = the input validation
# Validate and document the input received by the prediction endpoint.
class Features(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    price: float = Field(..., gt=0, description="Price per night, must be positive")
    minimum_nights: int = Field(..., ge=1, le=365, description="Minimum nights required for booking")
    number_of_reviews: int = Field(..., ge=0, description="Total number of reviews")
    reviews_per_month: float = Field(..., ge=0, description="Average reviews per month")
    calculated_host_listings_count: int = Field(..., ge=0, description="Number of listings by this host")
    availability_365: int = Field(..., ge=0, le=365, description="Days available out of 365")
    neighbourhood_group: str = Field(..., min_length=1, description="Borough or neighbourhood group")
    neighbourhood: str = Field(..., min_length=1, description="Specific neighbourhood name")


# Provide a simple endpoint to verify that the API is running.
@app.get('/')
def greet():
    return "Hello Guyss"

# Predict the room type for a single Airbnb listing.
@app.post('/predict')
def predict(features: Features):

    # Convert validated input into the feature structure expected by the model.
    row = pd.DataFrame([features.model_dump()], columns=COLUMNS)

    # Run prediction using the saved preprocessing and model pipeline.
    prediction  = model.predict(row)
    probability = model.predict_proba(row)

    # Return a frontend-friendly JSON response.
    return {
        "Predicted_room_type": prediction[0],
        "Probability": probability.tolist()[0]}