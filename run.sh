sudo docker build -t fb .
sudo docker rm myapp
sudo docker run --name myapp --net=host -p 8000:8000 --gpus all fb:latest 

