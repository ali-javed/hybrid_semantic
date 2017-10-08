# hybrid_semantic

This repository contains all source codes and datasets, including the gold standards, used in the Hybrid Semantic Clustering project. 

Direction for using the resources:
1. Pull all the source codes.
2. Pull dummy data files for testing "synth_data" located in the /data folder. 
3. Once all your codes and data are placed in the same structure as in the repository, execute main.py.

Main.py calls multiple scripts. At the end, you will see a cluster_assign folder in the main directory, which will contain hashtags and the cluster number those hashtags are assigned to. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gold Standards for all experiments can be found in the folder archived in data_set_and_gold_standard.zip.
There are three parameters that need to be set in metadata_based_distances.

1. Tweets file. 

a) For running GT-R1,GT-R2, GT-R3. GT-S1, GT-S2, GT-S3, GT-ALL, set tweet_file = "/data/combined.txt" in line 26.

b) hashtag_extracted_file_name = '/data/GT_Hashtags/<desired gt file here>.txt'

c) make sure ext_hashtags() is uncommented in line 274
  
if you want to run anyother hashtag file

a) set tweet_file = "/data/<desired file containing tweet in each line>.txt" in line 26.

b) uncomment line 31 for a generic hashtag file name

c) uncomment ext_hashtags() in line 274


Distance parameters for extracting flat clusters need to be set in form_text_and_metadata_based_clusters.py and Hybrid.py

