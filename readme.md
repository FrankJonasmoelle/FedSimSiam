### To run the python project locally, execute the following steps

To set up the virtual environment, run
```
    python3 -m venv .simsiam-venv
```

To activate the virtual environment, run
```
    source .simsiam-venv/bin/activate
```

To install the requirements, run
```
    pip3 install -r requirements.txt
```

Run it with
```
python3 train_simsiam.py --epochs 50 --lr 0.03 --momentum 0.9 --weight_decay 0.0005 --output_path 'simsiam.pth'
```

### To run the project in Docker, use the following commands
```
    docker build -t federated_simsiam .
```
then
```
    docker run --shm-size 8G --gpus all federated_simsiam
```

