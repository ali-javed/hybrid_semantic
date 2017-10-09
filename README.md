# hybrid_semantic

This repository contains all source codes and datasets, including the gold standards, used in the Hybrid Semantic Clustering project. 

Direction for using the resources:
1. Pull all the source codes.
2. Pull dummy data files for testing "synth_data" located in the /data folder. 
3. Once all your codes and data are placed in the same structure as in the repository, execute main.py.

Main.py calls multiple scripts. At the end, you will see a cluster_assign folder in the main directory, which will contain hashtags and the cluster number those hashtags are assigned to. 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gold Standards for all experiments can be found in the folder archived in data_set_and_gold_standard.zip.

There are two parameters that need to be set in the script metadata_based_distances.py: tweet file name and hashtag file name.  Here come instructions to set the parameter values in the script.  

1. Tweet file name:
Set tweet_file = "/data/combined.txt" in line 26.

2. Hashtag file name:
Set hashtag_extracted_file_name = '/data/GT_Hashtags/gt-file.txt' in line ???. (Here, gt-file is the ground truth hashtag file you want to use, e.g., GT-R1, GT-R2, GT-R3, GT-S1, GT-S2, GT-S3, GT-ALL.)

Note: if you want to use your own tweet data set and your own ground truth hashtag set, replace "combined.txt" above with your tweet dataset name and "gt-file.txt" above with your ground truth hashtag file name.

For any clustering task, the two parameters, "method" and "distance", for hierarchical clustering and flat cluster extraction need to be set in the scripts form_text_and_metadata_based_clusters.py (for metadata-based clustering) and Hybrid.py (for hybrid clustering).

