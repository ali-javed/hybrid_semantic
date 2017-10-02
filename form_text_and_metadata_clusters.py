# -*- coding: utf-8 -*-
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
import csv
from sklearn import metrics



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
         

path_variable = os.getcwd()

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


sense_similarity_matrix = np.zeros((len(unique_senses),len(unique_senses)))
counter1= 0
counter2 = 0
np.fill_diagonal(sense_similarity_matrix, 1, wrap=False)
count = 0
progress1 = 0
size_fine_grain = len(fine_grain_distances)
for row in fine_grain_distances:
    index1 = unique_senses.index(row.hashtag1+';'+row.word1+';'+str(row.sense1))
    index2 = unique_senses.index(row.hashtag2+';'+row.word2+';'+str(row.sense2))
    
    if sense_similarity_matrix[index1][index2] == 0:
        sense_similarity_matrix[index1][index2] = float_round(float(row[-1]),3,round)
        sense_similarity_matrix[index2][index1] = float_round(float(row[-1]),3,round)
    if (sense_similarity_matrix[index1][index2] != 0 and sense_similarity_matrix[index1][index2] != float_round(float(row[-1]),3,round) and unique_senses.index(row.hashtag1+';'+row.word1+';'+str(row.sense1)) != unique_senses.index(row.hashtag2+';'+row.word2+';'+str(row.sense2))) :
        print "Mismatch"
        print unique_senses.index(row.hashtag1+';'+row.word1+';'+str(row.sense1))
        print unique_senses.index(row.hashtag2+';'+row.word2+';'+str(row.sense2))
    
    count = count + 1
    progress =  100 * (float(count)/float(size_fine_grain))
    if (int(progress)%10 == 0 and int(progress)>int(progress1)):
        print "%i percent complete" %int(progress)
        progress1 = progress
    
    

distance_sense_matrix1 = 1 - sense_similarity_matrix  # convert simalirity matrix to dissimalirity matrix
sense_distance = np.around(distance_sense_matrix1, decimals = 3)


sense_distance.dump("saved_matrix/sense_distance_matrix.dat")

def write_excel(matrix,titles):
    fl = open('saved_matrix/semantic_sim_matrix.csv', 'w')

    writer = csv.writer(fl)
    writer.writerow(titles) #if needed
    for values in matrix:
        writer.writerow(values)

    fl.close()   



titles = []
for path, subdirs, files in os.walk(path_variable + '/virtual_documents'):
   for filename in sorted(files):
     f = filename
     if f != ".DS_Store":
        titles.append(filename)

     
    
 
 

tfidf_dist= np.load("saved_matrix/tfidf_distance.dat")
semantic_distance= np.load("saved_matrix/semantic_distance.dat")
sense_distance = np.load("saved_matrix/sense_distance_matrix.dat")


np.fill_diagonal(tfidf_dist, 0, wrap=False) #empty documents have distance 1
np.fill_diagonal(semantic_distance, 0, wrap=False) #some same concepts have distance greater than 0 in wn
np.fill_diagonal(sense_distance, 0, wrap=False)


observation_matrix_sense = ssd.squareform(sense_distance)
observation_matrix_semantic = ssd.squareform(semantic_distance)
observation_matrix_tfidf = ssd.squareform(tfidf_dist)      

#can use average
linkage_sense_semantic = hac.linkage(observation_matrix_sense, method = linkage_sense_semantic_method)                              
linkage_tfidf = hac.linkage(observation_matrix_tfidf, method = linkage_tfidf_method)
linkage_semantic = hac.linkage(observation_matrix_semantic, method = linkage_semantic_method)


   
flat_clusters_sense_semantic = hac.fcluster(linkage_sense_semantic, flat_clusters_sense_semantic_threshold, flat_clusters_sense_semantic_method)
flat_clusters_tfidf = hac.fcluster(linkage_tfidf, flat_clusters_tfidf_threshold, flat_clusters_tfidf_method)   #distance can be used
flat_clusters_semantic = hac.fcluster(linkage_semantic, flat_clusters_semantic_threshold, flat_clusters_semantic_method)    
          


silhouttee_sense = metrics.silhouette_score(sense_distance, flat_clusters_sense_semantic, metric='precomputed')         
silhouttee_word = metrics.silhouette_score(semantic_distance, flat_clusters_semantic, metric='precomputed')         


    
cluster_assignment_sense = dict(zip(unique_senses, flat_clusters_sense_semantic))
cluster_assignment_tfidf = dict(zip(titles, flat_clusters_tfidf))
cluster_assignment_semantic = dict(zip(titles, flat_clusters_semantic))


if not os.path.exists(path_variable + '/Cluster_assign'):
    os.makedirs(path_variable + '/Cluster_assign')

f = open("Cluster_assign/cluster_assignment_semantic.csv", "w")
f.close()
f = open("Cluster_assign/cluster_assignment_tfidf.csv", "w")
f.close()


with open("Cluster_assign/cluster_assignment_tfidf.csv", "a") as f:
    w = csv.writer(f,delimiter=';')
    w.writerow(('hashtag', 'cluster'))    # field header
f.close()

for (hashtag, cluster) in cluster_assignment_tfidf.iteritems():
    with open("Cluster_assign/cluster_assignment_tfidf.csv", "a") as f:
        f.write ('%s ; %i' % (hashtag, cluster))
        f.write("\n")
f.close()



with open("Cluster_assign/cluster_assignment_semantic.csv", "a") as f:
    w = csv.writer(f,delimiter=';')
    w.writerow(('hashtag', 'cluster'))    # field header
f.close()



for (hashtag, cluster) in cluster_assignment_semantic.iteritems():
    with open("Cluster_assign/cluster_assignment_semantic.csv", "a") as f:
        f.write ('%s ; %i' % (hashtag, cluster))
        f.write("\n")
f.close()

#for counting most common
L1 = list(flat_clusters_sense_semantic).count(1)


with open("Cluster_assign/cluster_assignment_semantic_senses.csv", "a") as f:
    w = csv.writer(f,delimiter=';')
    w.writerow(('hashtag', 'word', 'synset', 'cluster'))    # field header
f.close()

for (hashtag, cluster) in cluster_assignment_sense.iteritems():
    with open("Cluster_assign/cluster_assignment_semantic_senses.csv", "a") as f:
        if (list(flat_clusters_sense_semantic).count(cluster) > 0):
            f.write ('%s ; %i' % (hashtag, cluster))
            f.write("\n")
f.close()




tfidf_counter = Counter(flat_clusters_tfidf)
semantic_counter = Counter (flat_clusters_semantic)
semantic_sense_counter = Counter(flat_clusters_sense_semantic)

with open("Cluster_assign/sense_semantic_counter.csv", "w") as f:
    for k,v in  semantic_sense_counter.most_common():
        f.write( "{} {}\n".format(k,v) )
f.close()      

with open("Cluster_assign/tfidf_counter.csv", "w") as f:
    for k,v in  tfidf_counter.most_common():
        f.write( "{} {}\n".format(k,v) )
f.close()        
        
with open("Cluster_assign/semantic_counter.csv", "w") as f:
    for k,v in  semantic_counter.most_common():
        f.write( "{} {}\n".format(k,v) )
        
f.close()