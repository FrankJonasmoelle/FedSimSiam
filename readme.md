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

To train centralized SimSiam, run
```
python3 train_simsiam.py --epochs 800 --lr 0.03 --momentum 0.9 --weight_decay 0.0005 --output_path 'simsiam.pth'
```

To run its federated version, FedSimSiam, run
```
    python3 train_federation.py --num_clients 5 --alpha 0.5 --num_rounds 30 --local_epochs 25 --batch_size 64 --output_path 'fedsimsiam.pth'
```

To evaluate the trained model 'path_to_model.pth' on CIFAR-10, run 
```
    python3 evaluation_comparison.py --data_percentage 0.01 --epochs 100 --lr 0.003 --batch_size 256 --simsiam_path 'path_to_model.pth'
```


### To run the project in Docker, use the following commands
```
    docker build -t federated_simsiam .
```
then
```
    docker run --shm-size 8G --gpus all federated_simsiam
```

