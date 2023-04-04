# FROM python:3.10
FROM pytorch/pytorch

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/app/SimSiam"

CMD ["python", "/app/train_simsiam.py"]