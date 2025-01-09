
# Loan Default Prediction API

## Prerequisites

To run the code locally, you need to have the following installed:

- **Python 3.x**
- **Pip** (Python package manager)
- **Git** (optional, for cloning the repository)

## Step 1: Clone the Repository

Clone the repository to your local machine:

```bash
git clone https://github.com/SushainDevi/intern-Assignment-BRANCH.git
```

## Step 2: Install Required Libraries

Navigate to the project folder and install the required dependencies:

```bash
cd intern-Assignment-BRANCH
pip install -r requirements.txt
```

The `requirements.txt` file includes all the necessary libraries, including `pandas`, `numpy`, `fastapi`, `scikit-learn`, etc.

## Step 3: Run the API

To run the API, execute the following command:

```bash
uvicorn main:app --reload
```

This will start the FastAPI server on your local machine. You can access the API documentation at:

```
http://localhost:8000/docs
```

## Step 4: Make Predictions

You can use the `/predict` endpoint to make predictions by sending a POST request with the following data format:

```json
{
  "user_id": 123,
  "cash_incoming_30days": 50000,
  "gps_fix_count_scaled": 10.5,
  "movement_radius_approx_scaled": 0.75,
  "age": 30,
  "var_longitude": 0.002
}
```

## Step 5: Health Check

You can check if the API is running by visiting the `/health` endpoint:

```
http://localhost:8000/health
```

---

Let me know if you need further modifications!
