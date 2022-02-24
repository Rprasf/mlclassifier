from fastapi.testclient import TestClient
from main import app
import pandas as pd
from pathlib import Path
from http import HTTPStatus

client = TestClient(app)

#Todo : move ml folder inside api folder before testing.

def test_validate_models():
    response = client.get('/models')
    assert response.status_code == 200
    print(response.json())
    assert response.json() ==  ["RandomForestClassifier","sk-learn-random-forest-reg-model"]


    