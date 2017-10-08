# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import scipy.cluster.hierarchy as hac
import scipy.cluster as sc
import os
import matplotlib.pyplot as plt
import csv
from collections import Counter # to count hashtags
import scipy.spatial.distance as ssd
import collections
from itertools import chain
from math import ceil, floor


#set methods and thresholds for clustering
linkage_sense_semantic_method = 'average'  
linkage_tfidf_method = 'single'
linkage_semantic_method = 'average' 

flat_clusters_sense_semantic_threshold = 0.4
flat_clusters_tfidf_threshold = 0.4
flat_clusters_semantic_threshold = 0.4
         
flat_clusters_sense_semantic_method = 'distance'
flat_clusters_tfidf_method = 'distance'
flat_clusters_semantic_method = 'distance'
   
   
   
   

#simple semantic annotation will not work to make it work, need to take all titles i.e. selected hashtags.
def getKey(item):
     return item.split(";")[2]

def float_round(num, places = 0, direction = floor):
    return direction(num * (10**places)) / float(10**places)


with open('saved_matrix/fine_grain_distances.csv') as infile:
          reader = csv.DictReader(infile)
          Data = collections.namedtuple('Hashtag_sense_distances', reader.fieldnames)
          fine_grain_distances = [Data(**row) for row in reader]

print len(fine_grain_distances)
#calculate how many unique senses
sense1_list = sorted(set(list(chain.from_iterable([[fine_grain.hashtag1+';'+fine_grain.word1+';'+str(fine_grain.sense1)] for fine_grain in fine_grain_distances]))))
sense2_list = sorted(set(list(chain.from_iterable([[fine_grain.hashtag2+';'+fine_grain.word2+';'+str(fine_grain.sense2)] for fine_grain in fine_grain_distances]))))


unique_senses_full_sort = sorted(set(sense1_list+sense2_list))

unique_senses = sorted(unique_senses_full_sort, key = getKey)

print len(unique_senses)
#sense1_len = len(sorted(set(list(chain.from_iterable([[fine_grain.sense1] for fine_grain in fine_grain_distances])))))
#sense2_len = len(sorted(set(list(chain.from_iterable([[fine_grain.sense2] for fine_grain in fine_grain_distances])))))


def write_excel(matrix,titles):
    fl = open('saved_matrix/semantic_sim_matrix.csv', 'w')

    writer = csv.writer(fl)
    writer.writerow(titles) #if needed
    for values in matrix:
        writer.writerow(values)

    fl.close()   


path_variable = os.getcwd()
titles = []
for path, subdirs, files in os.walk(path_variable + '/virtual_documents'):
   for filename in sorted(files):
     f = filename
     if f != ".DS_Store":
        titles.append(filename)

     
    
 
 

tfidf_dist= np.load("saved_matrix/tfidf_distance.dat")
sense_distance = np.load("saved_matrix/sense_distance_matrix.dat")
#sense_distance = np.load("saved_matrix/semantic_distance.dat")  #uncomment for comparision with semantic


np.fill_diagonal(tfidf_dist, 0, wrap=False) #empty documents have distance 1
np.fill_diagonal(sense_distance, 0, wrap=False)


observation_matrix_sense = ssd.squareform(sense_distance)
observation_matrix_tfidf = ssd.squareform(tfidf_dist)      

linkage_sense_semantic = hac.linkage(observation_matrix_sense, method = linkage_sense_semantic_method)                              
linkage_tfidf = hac.linkage(observation_matrix_tfidf, method = linkage_tfidf_method)




       
flat_clusters_sense_semantic = hac.fcluster(linkage_sense_semantic, flat_clusters_sense_semantic_threshold, flat_clusters_sense_semantic_method)
flat_clusters_tfidf = hac.fcluster(linkage_tfidf, flat_clusters_tfidf_threshold, flat_clusters_tfidf_method)   #distance can be used

cluster_assignment_sense = dict(zip(unique_senses, flat_clusters_sense_semantic))
cluster_assignment_tfidf = dict(zip(titles, flat_clusters_tfidf))



tfidf_counter = Counter(flat_clusters_tfidf)
semantic_sense_counter = Counter(flat_clusters_sense_semantic)


#write files
#no need to write files again if clusterize sep trace has already run
'''
f = open("Cluster_assign/cluster_assignment_semantic_senses.csv", "w")
f.close()
f = open("Cluster_assign/cluster_assignment_tfidf.csv", "w")
f.close()


with open("Cluster_assign/sense_semantic_counter.csv", "w") as f:
    for k,v in  semantic_sense_counter.most_common():
        f.write( "{} {}\n".format(k,v) )
      

with open("Cluster_assign/tfidf_counter.csv", "w") as f:
    for k,v in  tfidf_counter.most_common():
        f.write( "{} {}\n".format(k,v) )
        

for (hashtag, cluster) in cluster_assignment_tfidf.iteritems():
    with open("Cluster_assign/cluster_assignment_tfidf.csv", "a") as f:
        f.write ('%s ; %i' % (hashtag, cluster))
        f.write("\n")



#for counting most common
L1 = list(flat_clusters_sense_semantic).count(1)

for (hashtag, cluster) in cluster_assignment_sense.iteritems():
    with open("Cluster_assign/cluster_assignment_semantic_senses.csv", "a") as f:
        if (list(flat_clusters_sense_semantic).count(cluster) > 1):
            f.write ('%s ; %i' % (hashtag, cluster))
            f.write("\n")


'''


'''
#hac.cophenet(linkage_tfidf,observation_semantic)
'''



tfidf_counter = Counter(flat_clusters_tfidf)
semantic_sense_counter = Counter(flat_clusters_sense_semantic)



#if matrix is present start comment here



indexes1 = []
indexes2 = []
counter1 = 0
counter2 = 0
matrix = np.zeros((len(titles),len(titles)))
cluster_counter = 1
progress1 = 0
for title1 in sorted(titles):
    indexes1 =[value for key, value in cluster_assignment_sense.items() if title1 in key.lower()]
    
    for title2 in sorted(titles):
        cluster_counter = 2
        indexes2 =[value for key, value in cluster_assignment_sense.items() if title1 in key.lower()]
        if title1 != title2:
            cluster_counter = 0
            if len(set ([value for key, value in cluster_assignment_sense.items() if title1 in key.lower()]) & set ([value for key, value in cluster_assignment_sense.items() if title2 in key.lower()])):
                cluster_counter = cluster_counter + 1
                
            if cluster_assignment_tfidf[title1] == cluster_assignment_tfidf[title2]:
                cluster_counter = cluster_counter + 1 
            
        matrix[counter1][counter2] = (cluster_counter/2)
        
        
        counter2 = counter2 + 1
    
        
                                               
    counter1= counter1+1
    counter2 = 0
    print (counter1)   
   

dist_hybrid = 1- matrix

dist_hybrid.dump("saved_matrix/dist_hybrid.dat")

#if matrix is present end comment here



dist_hybrid= np.load("saved_matrix/dist_hybrid.dat")
observation_matrix_hybrid = dist_hybrid[np.triu_indices(len(dist_hybrid),1)]

linkage_hybrid = hac.linkage(observation_matrix_hybrid,method = 'average')
  
flat_clusters_hybrid = hac.fcluster(linkage_hybrid, 0.5, 'distance')  #distance can be used

cluster_assignment_hybrid = dict(zip(titles, flat_clusters_hybrid))


with open("Cluster_assign/cluster_assignment_hybrid.csv", "a") as f:
    w = csv.writer(f,delimiter=';')
    w.writerow(('hashtag', 'cluster'))    # field header
f.close()


for (hashtag, cluster) in cluster_assignment_hybrid.iteritems():
    with open("Cluster_assign/cluster_assignment_hybrid.csv", "a") as f:
        f.write ('%s ; %i' % (hashtag, cluster))
        f.write("\n")
f.close()


hybrid_counter = Counter(flat_clusters_hybrid)
with open("Cluster_assign/hybrid_counter.csv", "w") as f:
    for k,v in  hybrid_counter.most_common():
        f.write( "{} {}\n".format(k,v) )
f.close()
