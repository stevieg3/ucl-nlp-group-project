A few steps are needed to be able to run the code:

1) download the AG News dataset. The original is available from Kaggle in csv format. In order to use it with AllenNLP, I had to convert it to JSON/JSONL. You can download those files from these links:
	- train set: https://drive.google.com/file/d/15TH-GUaH4WOiJaARI_PIzM2Jvdqanx1O/view?usp=sharing
	- val set: https://drive.google.com/file/d/1Ck8QOXg7ulQBlqzuseAD5smzZfaBGKmb/view?usp=sharing
	- test set: https://drive.google.com/file/d/1xFsmsi4zCrhNO5uRm-VNk1qd1b7wlHXG/view?usp=sharing

2) set up the AllenNLP environment:
my recommendation would be to upload a copy of the folder called 'AllenNLP' to a directory on your Google Drive. The directory consists of the following:
	- bcn-running_config: this is the notebook that you need to run
	- config_BCN: the congif file that brings the model and the dataset reader together
	- tagging: this includes the code for the dataset reader
	- BCN_model: this includes the code for the BCN model from AllenNLP

3) You then just need to update a few links to make sure that it all runs on your machine:
a) in the config file: update the links to the train and val sets (to wherever you saved them)
b) in the notebook: 
- the first cell should point to the directory that corresponds to the 'AllenNLP' folder (i.e. the one where the config file is saved)
- the second cell should have a link to wherever you saved the test set

