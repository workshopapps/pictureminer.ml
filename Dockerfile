FROM tiangolo/uvicorn-gunicorn:python3.8-slim

RUN mkdir /fastapi

WORKDIR /fastapi
COPY requirements.txt /fastapi


RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt


COPY . /fastapi

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]