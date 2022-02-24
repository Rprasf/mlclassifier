FROM python:3.9-slim

RUN mkdir /api
RUN mkdir /ml


COPY ./api/requirements.txt /api/requirements.txt
RUN pip install -r api/requirements.txt

COPY ./api /api
COPY ./ml /ml


EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
    
