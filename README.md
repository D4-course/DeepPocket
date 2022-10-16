# DeepPocket: Ligand Binding Site Detection and Segmentation using 3D Convolutional Neural Networks

## Requirements: 
- Docker
- CUDA >11.6

## Installation Instructions:
- Open a terminal and clone the repo
```
git clone https://github.com/D4-course/DeepPocket.git
```
- Download [trained models](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/rishal_aggarwal_research_iiit_ac_in/EoJSrvuiKPlAluOJLjTzfpcBT2fVRdq8Sr4BMmil0_tvHw?e=Kj7reS) and load them into ```DeepPocket``` folder in the parent repo. You can download any classified_models and segmentation_models for this purpose or train one yourself and use. (Refer to [manual](./MANUAL.md))
- Execute ```run.sh``` that will build the docker image, run a container where the frontend and backend will be executed automatically
```
./run.sh
or
sh run.sh
```
- Navigate to the network or external URL that will be displayed on the terminal
```
(For Refernce only)
You can now view your Streamlit app in your browser.

  Network URL: http://10.42.0.88:8501
  External URL: http://10.1.34.46:8501
```
## Website Instructions
