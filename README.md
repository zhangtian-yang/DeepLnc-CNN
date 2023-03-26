DeepLnc-CNN
====
Description:
------------
DeepLnc-CNN is able to identify the long non-coding RNAs in human and mouse.

Installation:
-------------
- <span  style="color: #5bdaed; font-weight: bold">python3.8</span>
- pytorch==1.8.2+cpu
- numpy==1.21.5
``` 
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
pip3 install numpy==1.21.5
``` 
``` 
pip install -r yours/requirement.txt
``` 
Optional arguments:
-------------------
```
  -h, --help            Show this help message and exit.
  --addresses ADDRESSES
                        Tianyang.Zhang819@outlook.com
  -i INPUTFILE, --inputFile INPUTFILE
                        -i input.txt (The input file is a complete Fasta
                        format sequence.)
  -o OUTPUTFILE, --outputFile OUTPUTFILE
                        -o output_prediction.html (Results of predicting 
                        lncRNAs are saved under results folder.)
  -s SPECIES, --species SPECIES
                        -s Human/Mouse (Choose one from two species to
                        use.)
  -ts THRESHOLD, --threshold THRESHOLD  
                        -ts 0.5(Prediction result threshold)
```
Example:
--------
```
python DeepLnc-CNN.py -i Example.txt -o output.html -s Human -ts 0.5
```
***
Version number：V0.1.0 <br>
Updated date：2023-03-26 <br>
Email: Tianyang.Zhang819@outlook.com
***
