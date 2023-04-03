# FROM python:3.10
FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu20.04

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app/SimSiam"

CMD ["python", "/app/train_simsiam.py"]