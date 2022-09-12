# DeepPocket

## Steps to fork Fpocket
```
git clone https://github.com/Discngine/fpocket.git;
cd fpocket;
make ;
sudo make install;
```

## Building docker image
```
docker build -t dp .
```

## Creating the docker container 
```
docker run -it --name myapp --rm --volume $(pwd):/usr/src/app --net=host dp:latest sh
```
*Note: This uses the GPUs available and also uses the pwd where the docker container is being executed.

## Predicting the centers of given protein file
```
python predict.py -p protein.pdb -c first_model_fold1_best_test_auc_85001.pth.tar -s seg0_best_test_IOU_91.pth.tar -r 3
```
