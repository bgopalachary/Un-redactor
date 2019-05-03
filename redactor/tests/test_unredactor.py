
import glob
import ipytest
import io
import os
import pdb
import sys

from ipynb.fs.full.Unredactor import main
import ipynb.fs.full.Unredactor

glob_text2= "D:/DSA/Text Analytics/Project 2/testtest/*.txt"

def test_main():
    a,b,c,d=main()
    assert a is not None
    assert type(b) == list
    assert len(c) > 0 

def test_training():
    Mod=ipynb.fs.full.Unredactor.trainingpart()
    assert Mod is not None
    
    
def test_testing():
    for thefile in glob.glob(glob_text2):
        with io.open(thefile, 'r', encoding='utf-8',errors='ignore') as fyl:
            text = fyl.read()
            xtest=ipynb.fs.full.Unredactor.testingpart(text)
            assert type(xtest) is tuple
            assert len(xtest)>0
    
    

    

