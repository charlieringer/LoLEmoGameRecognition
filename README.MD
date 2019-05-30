## Multimodal Joint Emotion and Game Context Recognition in League of Legends Livestreams
This repository contains the code to support the paper "Multimodal Joint Emotion and Game Context Recognition in League of Legends Livestreams" presented at the 2019 IEEE Conference on Games. 

## Data
The assositated dataset can be downloaded from: 

Copy all files/folders found there into this directory

## Usage
This data set needs to be convered into npy files first. This can be done with `utils/process_data_set.py`. Once that is complete `trainer.py` will run the full set of expeirments. To apply the oversampling technique detailed in the paper run `augment_annotations.py` from the utils folder.

## Requirmentments
Requirements can be found on `requirements.txt`.

## Cite this work
Please cite this work as:
`Ringer, Charles, Walker, James Alfred, Nicolaou, Mihalis A. (2019) Multimodal Joint Emotion and Game Context Recognition in League of Legends Livestreams, In: Proceedings of the IEEE Confernce on Games 2019. IEEE`

## Licence