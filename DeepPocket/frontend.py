# import the streamlit library
import streamlit as st
import py3Dmol
from pathlib import Path
from stmol import *
import json
import requests
import pandas as pd
import zipfile
import os


# Base URL for API Calls
BASE_URL="http://localhost:8000/api/v1/"

# give a title to our app
st.title('Welcome to DeepPocket Tool')
# Upload Protien File
protien_file = st.file_uploader("Upload Input Protein", type=["pdb"])

# If SEGMENTATION Required
seg_status = st.checkbox('Segment the centers?')

# compare status value
if(seg_status):
    # Enter the no of centers to segment
    num_pockets = st.number_input("Enter the number of centers to segment",min_value=1)


# function handler for displaying the protein molecules
def render_mol(xyz):
    style = st.selectbox('Protein Style',['line','cross','stick','sphere','cartoon','ribbon'])
    xyzview = py3Dmol.view(width=300,height=300)
    xyzview.addModel(xyz,'pdb')
    xyzview.setStyle({style:{'color':'spectrum'}})
    xyzview.setBackgroundColor('white')#('0xeeeeee')
    xyzview.addSurface(py3Dmol.SES,{'opacity':0.8,'color':'spectrum'})
    xyzview.zoomTo()
    showmol(xyzview, height = 300,width=300)

# Split the page into two columns
col1, col2 = st.columns(2)
# Create Submit button
var1 = st.empty()
predict = var1.button('Predict')

# check if the button is pressed or not
if(predict):
    if(protien_file):
        st.success("Request Submitted")
        # Upload the protein file to backend
        files = {"file": protien_file.getvalue()}

        with st.spinner("Please Wait.."):
            if(seg_status):
                # Raise the post request
                res = requests.post(f"{BASE_URL}segment/num_pockets/{num_pockets}", files=files)
                # Seperate .csv and .zip files from obtained
                open('temp.zip', 'wb').write(res.content)
                with zipfile.ZipFile("temp.zip","r") as zip_ref:
                    zip_ref.extractall(".")
                st.download_button('Click to download output pocket file', res.content, file_name='segmented_pocket_centers.zip')
            else:
                res = requests.post(f"{BASE_URL}rank/", files=files)
                open('temp.csv', 'wb').write(res.content)
                df = pd.read_csv('temp.csv',names=["x coordinate", "y coordinate", "z coordinate", "Binding Site Probability"])

# Display the probabilities in 2nd column
        with col2:
            if(seg_status):
                df = pd.read_csv('pocket_locations.csv',names=["x coordinate", "y coordinate", "z coordinate", "Binding Site Probability"])
                for file in os.listdir():
                    if file.endswith('.dx') or file.endswith('.csv'):
                        os.remove(file)
            else:
                df = pd.read_csv('temp.csv',names=["x coordinate", "y coordinate", "z coordinate", "Binding Site Probability"])
            col2.write(df)
            col2.text("List of centers returned by the algorithm: (sorted in decreasing order of probabilities)")
    else:
        st.error("Please Input all the files")
#    st.experimental_rerun()
# Render protein file
if protien_file:
    with col1:
        xyz = protien_file.getvalue().decode("utf-8")
        render_mol(xyz)
        st.text("Interactive input protein modelcule")


