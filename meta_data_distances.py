from __future__ import division
import wikipedia
from nltk.corpus import wordnet as wn
import os
from nltk.tag import pos_tag
import wordsegment
import pdb
from collections import Counter # to count hashtags
import re #to get words from filtered_tweets
import numpy as np
import scipy.cluster.hierarchy as hac
import sys
from itertools import chain
import matplotlib.pyplot as plt
import collections
import cPickle
import collections
import csv
import os



#variables

path_variable = os.getcwd()
tweet_file = "/data/synth_data.txt"
minimum_tweet_limit = 0   #if you want to drop hashtags with less than x tweets, set x here.
hashtag_extracted_file_name = '/hashtags.txt'

#function definitions


def is_ascii(s):
    return all(ord(c) < 128 for c in s)

hashtag_sense_distance = collections.namedtuple("Hashtag_Sense_Distance", ['hashtag1', 'hashtag2','word1','word2', 'sense1','sense2', 'similarity'])

hashtag_finegrain = collections.namedtuple("Hashtag_finegrain", ['hashtag', 'word', 'sense'])
center_distance = collections.namedtuple('Center_Distance', ['center', 'distance'])



def ext_hashtag():
    filtered_tweets_file = path_variable + tweet_file

    
    num_line = 0

    with open(filtered_tweets_file) as f:
        hashtags = []
        for line in f:
            matches = re.findall(r'(?<!\w)(\#\w+)', line.lower())
            hashtags.extend(matches)
            num_line = num_line +1
         

    unique_hashtags = sorted(set(hashtags))
    hashtags_occurence = Counter(hashtags)
    print len(unique_hashtags)
    print num_line
    hashtags_extracted = open(path_variable + hashtag_extracted_file_name, 'w') #make document listing hashtags
    for item in sorted(unique_hashtags):
        hashtags_extracted.write("%s\n" % item)
    
    hashtags_extracted.close()



def getWNConcept(hashtag):
    hashtag = hashtag.lower()
    fine_grain = None
    synsets = wn.synsets(hashtag)
    if synsets:
        fine_grain = hashtag_finegrain(hashtag = hashtag, word = hashtag, sense = synsets)
    
    
    if not synsets:
        
        word_list = wordsegment.segment(hashtag)
        length = len(word_list)
        for x in range (0,length):
            word_list.pop(0)
            new_hash = "".join(word_list)
            synsets = wn.synsets(new_hash)
            
            if synsets:
                fine_grain = hashtag_finegrain(hashtag = hashtag, word = hashtag, sense = synsets)
                break
        
    concepts = []            
    for synset in synsets:
        concepts.append(synset)
                
    return fine_grain

def getWikipediaCandidates(hashtag):
    nouns_hashtag = []
    nouns_all_wiki = []
    wordnet_concepts = []
    concept = []
    all_concepts = []
    list_of_hidden_cat = ['wikipedia', 'articles', 'wikidata']
    wikicategory = None
    enter = 1;
    try:
        wikicategory = wikipedia.page(title = hashtag, auto_suggest = False, redirect = False).categories
    except:
        print "exception entered"
        print hashtag
        
       
    if wikicategory:
        print "wikicatagory found"
        print wikicategory
        for concepts in wikicategory:
            if any(word.lower() in concepts.lower() for word in list_of_hidden_cat):
                pass
                
            else:
                tagged_sent = pos_tag(concepts.split())
                nouns_hashtag.extend([word for word,pos in tagged_sent if pos.startswith('NN')])
                nouns_all_wiki.extend(nouns_hashtag)
    
        unique_nouns = set(nouns_all_wiki)
        for noun in unique_nouns:
            concept = None
            try:
                word_list = wordsegment.segment(noun)
            except:
                enter = 0;
            
            if enter ==1:
                concept = getWNConcept(noun)
            
            if concept:
                concept_correct_hash = hashtag_finegrain(hashtag = hashtag, word = concept.word, sense = concept.sense)
                all_concepts.append(concept_correct_hash)


                        
    
    return all_concepts


def getcandidate(h):
    
    candidates = getWNConcept(h)
    
    if (candidates):
        return candidates
    else:
        candidates = getWikipediaCandidates(h)
    return candidates
    

def makevirtual(hashtag):
    tweet_counter = 0;
    
    filtered_tweets_file = path_variable + tweet_file
    hashtag_file = open(path_variable + "/virtual_documents" + "/" + hashtag, 'w') #make virtual document for hashtags
    
    with open(filtered_tweets_file) as f: #for virtual document
        
        for line in f:
            matches = re.findall(r'(?<!\w)(\#\w+)', line.lower())
            if hashtag.strip().lower() in matches:
                  
                hashtag_file.write(line)
                tweet_counter = tweet_counter +1;
                    
    hashtag_file.close() 
    if (tweet_counter < minimum_tweet_limit):
        os.remove(hashtag_file.name)
        return 1;
        
    return 0   
    
def getCandidates(hashtag):
    hashtag.join("#")
    index = selected_hashtags.index(hashtag)
    index_fine_grain = hashtag_noun_sense.index(hashtag == hashtag) 
    return hashtag_concepts[index]
    
    


def semantic_simalirity(h1,h2):
    
    fine_grain_dist = None
    #L1 = getCandidates(h1)
    #L2 = getCandidates(h2)
    L1 = [[finegrain.sense] for finegrain in hashtag_noun_sense if finegrain.hashtag.lower() ==h1[1:].lower()]
    L2 = [[finegrain.sense] for finegrain in hashtag_noun_sense if finegrain.hashtag.lower() ==h2[1:].lower()]
    Noun1 = [[finegrain.word] for finegrain in hashtag_noun_sense if finegrain.hashtag.lower() ==h1[1:].lower()]
    Noun2 = [[finegrain.word] for finegrain in hashtag_noun_sense if finegrain.hashtag.lower() ==h2[1:].lower()]
    
    try:
        Noun1 = list(chain.from_iterable(Noun1))
        
        Noun2 = list(chain.from_iterable(Noun2))
       
    except:
        Noun1 = Noun1
        Noun2 = Noun2
    
    
    try:
        LCh1 = list(chain.from_iterable(L1))
    except:
        LCh1 = L1
    try:   
        LCh2 = list(chain.from_iterable(L2))
    except:
        LCh2 = L2
        
    try:
        LCh1 = list(chain.from_iterable(LCh1))
    except:
        LCh1 = LCh1
    try:   
        LCh2 = list(chain.from_iterable(LCh2))
    except:
        LCh2 = LCh2    
    
    if len(LCh1) ==0 or len(LCh2) == 0:
        return 0
        
  
    
    simmax = 0
    sim = 0
    for concepts1 in LCh1:
         for concepts2 in LCh2: 
            try:
                
                sim = concepts1.wup_similarity(concepts2)
                
                if sim>simmax: #simalirity 
                    simmax = sim
                    concept1_final = concepts1
                    concept2_final = concepts2
                    fine_grain_dist = hashtag_sense_distance(hashtag1 = h1,hashtag2 = h2, word1 = Noun1,word2 = Noun2, sense1 = concepts1,sense2 = concepts2, similarity = simmax)
                    
            except:
                print "exception entered in semantic similarity  exiting..."
                sys.exit()
                
    for concepts2 in LCh2:
     for concepts1 in LCh1: 
        try:

            sim = concepts2.wup_similarity(concepts1)
            if sim>simmax: #simalirity 
                simmax = sim
                concept1_final = concepts1
                concept2_final = concepts2
                fine_grain_dist = hashtag_sense_distance(hashtag1 = h1,hashtag2 = h2, word1 = Noun1,word2 = Noun2, sense1 = concepts1,sense2 = concepts2, similarity = simmax)
        except:
            print "exception occured in sementic similarity, exiting.."
            sys.exit()          
   
   
    if (simmax ==0):
        fine_grain_dist = hashtag_sense_distance(hashtag1 = h1,hashtag2 = h2, word1 = Noun1,word2 = Noun2, sense1 = 'no similarity returned',sense2 = 'no similarity returned', similarity = 0)
    
    return fine_grain_dist
    
    
########################## MAIN ######################

ext_hashtag()  #no need to extract hashtags again and again

if not os.path.exists(path_variable + '/hashtag_and_senses'):
    os.makedirs(path_variable + '/hashtag_and_senses')

if not os.path.exists(path_variable + '/virtual_documents'):
    os.makedirs(path_variable + '/virtual_documents')

  
extracted_hashtags_file = open(path_variable + hashtag_extracted_file_name, 'r')
extractedHashtags = extracted_hashtags_file.read().splitlines()


hashtag_concepts= []
selected_hashtags = []
hashtag_concept_sense = []
hashtag_noun_sense = []
counter1 = 0
counter2 = 0
removed_files = 0

for hashtag in extractedHashtags:
    candidates =[]
    candidates = getcandidate(hashtag[1:])   # get candidates

    
    
    if candidates:
        if len(np.shape(candidates)) > 1:
            for candidate in candidates:
                hashtag_noun_sense.append(candidate)
        else:
            hashtag_noun_sense.append(candidates)
        hashtag_concepts.append([])#save a list of concepts
        virtual_document = open(path_variable + '/hashtag_and_senses' + "/" + hashtag, 'w') #make virtual document for hashtags
        virtual_document.write("%s\n" %hashtag)
        
        for candidate in candidates:  
            virtual_document.write("%s\n" % str(candidate))
            hashtag_concepts[counter1].append(candidate)   
            
        
        virtual_document.close()
        counter1 = counter1+1
        selected_hashtags.append(hashtag)
        value = makevirtual(hashtag)
        removed_files = removed_files + value
    
print ("Total virtual documents removed because of not having 5 tweets = ", removed_files)    

print ("Virtual documents created")        

# calculate simalirity matrix

semantic_simalirity_dic_matrix ={}
semantic_simalirity_matrix =[]
matrix = np.zeros((len(selected_hashtags),len(selected_hashtags)))


selected_hashtags = sorted(selected_hashtags)

fine_grain_distances = []
dummy_fine_grain_distance = hashtag_sense_distance('','','','','','','')


counter1= 0
counter2 = 0
counter3 = 0

for hashtag1 in selected_hashtags:
    semantic_simalirity_matrix.append([])
    for hashtag2 in selected_hashtags:
        fine_grain_distances.append(dummy_fine_grain_distance)
        sim = semantic_simalirity(hashtag1,hashtag2)
        semantic_simalirity_matrix[counter1].append(sim.similarity)
        fine_grain_distances[counter3] = fine_grain_distances[counter3]._replace(hashtag1 = sim.hashtag1, hashtag2 = sim.hashtag2, word1 = sim.word1[0], word2 = sim.word2[0], sense1 = sim.sense1, sense2 = sim.sense2, similarity = sim.similarity)
        matrix[counter1][counter2] = sim.similarity
        counter2 = counter2 + 1
        counter3 = counter3 + 1
        semantic_simalirity_dic_matrix.update({(hashtag1,hashtag2):sim})
    counter1= counter1+1
    counter2 = 0
    print (counter1)
    
dis_matrix1 = 1 - matrix  # convert simalirity matrix to dissimalirity matrix
dis_matrix = np.around(dis_matrix1, decimals = 3)


if not os.path.exists(path_variable + '/saved_matrix'):
    os.makedirs(path_variable + '/saved_matrix')
    
dis_matrix.dump("saved_matrix/semantic_distance.dat")



#calculate how many unique senses
sense1_list = sorted(set(list(chain.from_iterable([[str(fine_grain.sense1)] for fine_grain in fine_grain_distances]))))
sense2_list = sorted(set(list(chain.from_iterable([[str(fine_grain.sense2)] for fine_grain in fine_grain_distances]))))

unique_senses = sorted(set(sense1_list+sense2_list))
print len(unique_senses)

sense_similarity_matrix = np.zeros((len(unique_senses),len(unique_senses)))
counter1= 0
counter2 = 0


with open('saved_matrix/fine_grain_distances.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(('hashtag1', 'hashtag2', 'word1', 'word2','sense1','sense2','similarity'))    # field header
    w.writerows([(data.hashtag1, data.hashtag2, data.word1, data.word2, data.sense1, data.sense2, data.similarity) for data in fine_grain_distances])
f.close()    
    
hashtags_selected_file = open("saved_matrix/selected_hashtags.txt", 'w') #make document listing hashtags    
for item in sorted(selected_hashtags):
        hashtags_selected_file.write("%s\n" % item)   
hashtags_selected_file.close()