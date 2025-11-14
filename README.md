This project currently contains the following items:

- data_handling/ includes functionality to preprocess raw data. functions in this folder are called by mistr/
- mistr/ from https://github.com/malradhi/MiSTR that includes files to both embed brain data AND use a transformer to decode predicted audio waveforms from the neural data. mistr/ uses files from data_handling/ to preprocess data
- results/ includes predicted .npy files after running mistr/.. this will be used as our baseline
