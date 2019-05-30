## Multimodal Joint Emotion and Game Context Recognition in League of Legends Livestreams
This repository contains the code to support the paper "Multimodal Joint Emotion and Game Context Recognition in League of Legends Livestreams" presented at the 2019 IEEE Conference on Games. 

## Data
The associated dataset can be downloaded from: 

Copy all files/folders found there into this directory

## Usage
- The data set needs to be converted into npy files first. This can be done with `utils/process_data_set.py`. 
- (optional) To apply the oversampling technique detailed in the paper run `utils/augment_annotations.py`.
- Once these preprocessing steps are complete `trainer.py` will run the full set of experiments. 

## Requirements
Requirements can be found in `requirements.txt`. These can be installed with `pip install -r requirements.txt`.

## Cite this work
Please cite this work as:
`Ringer, Charles, Walker, James Alfred, Nicolaou, Mihalis A. (2019) Multimodal Joint Emotion and Game Context Recognition in League of Legends Livestreams, In: Proceedings of the IEEE Conference on Games 2019. IEEE`

## Licence