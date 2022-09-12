from re import A
import sys
import os
from tkinter import E
import torch
sys.path.append( '../DeepPocket/')
import clean_pdb
import get_centers
from model import Model
import predict

#  CleanPdbTest
def test_clean_pdb():
    '''check the output of not-existed pdb file'''
    assert clean_pdb.clean_pdb('tests/data/1A0S.pdb','tests/data/1A0S_nowat.pdb') is None

def test_clean_pdb_1():
    '''check the output of existed pdb file'''
    assert clean_pdb.clean_pdb('../tests/protein.pdb','../tests/protein_nowat.pdb') is not None

#  Get_center_test
def test_get_centers():
    '''check the output of not-existed directory'''
    try:
        get_centers.get_centers('tests/data/1A0S')
    except AssertionError:
        assert True
    else:
        assert False

def test_get_centers_1():
    '''check the output of existed directory'''
    try:
        get_centers.get_centers('../tests/data/pockets')
    except AssertionError:
        assert False
    else:
        if os.path.exists('../tests/data/pockets/bary_centers.txt'):
            assert True
        else:
            assert False

#  Model_test of CNN architecture
def test_model():
    '''check the output of CNN architecture'''
    modl = Model()
    assert modl is not None

#  Model_test of CNN architecture
def test_model_1():
    '''check the output of CNN architecture'''
    try:
        modl = Model()
        modl.train()
    except Exception:
        assert False
    else:
        assert True
    
# Predict_test
def test_get_model_gmaker_eprovider():
    '''check the output of get_model_gmaker_eprovider'''
    try:
        predict.get_model_gmaker_eprovider('tests/data/1A0S.types', 1, Model(),'sdf','sdfasd')
    except AssertionError:
        assert True
    else:
        assert False