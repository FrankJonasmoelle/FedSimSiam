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


### To run the project in Docker, use the following commands
```
    docker build -t federated_simsiam .
```
then
```
    docker run federated_simsiam
```

