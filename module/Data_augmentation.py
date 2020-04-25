#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Data_augmentation.py
#  
#  Copyright 2020 Yves <yves@mercadier>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  
import pandas as pd 
import numpy as np # linear algebra

##############
#augmentation
def word_shuffle(sentence):
    new_sentences = []
    words = word_tokenize(sentence)
    random.shuffle(words)
    new_sentences=' '.join(words)
    return new_sentences
    
def word_synonym(sentence):
	words = word_tokenize(sentence)
	n_sentence=sentence
	for w in words:
		if w not in stoplist:
			if len(wordnet.synsets(w))>0:
				synonym = wordnet.synsets(w)[0].lemma_names()[0]
				if w!=synonym:
					#print(w)
					#print(synonym)
					n_sentence = n_sentence.replace(w, synonym)  # we replace with the first synonym
	return n_sentence

def stat_token():
	max_token=0
	cpt_token=0
	for text in train_origin['text']:
		token=classification.tokenizer.tokenize(text)
		cpt_token+=len(token)
		if max_token<len(token):
			max_token=len(token)
	print("max token : {}".format(max_token))
	print("moyenne token : {}".format(cpt_token/len(train_origin['text'])))

#https://github.com/google-research/uda/blob/master/text/augmentation/word_level_augment.py
def get_data_stats(texts):
  """Compute the IDF score for each word. Then compute the TF-IDF score."""
  word_doc_freq = collections.defaultdict(int)
  # Compute IDF
  for text in texts:
    cur_word_dict = {}
    cur_sent = text.split(' ')
    for word in cur_sent:
      cur_word_dict[word] = 1
    for word in cur_word_dict:
      word_doc_freq[word] += 1
  idf = {}
  for word in word_doc_freq:
    idf[word] = math.log(len(texts) * 1. / word_doc_freq[word])
  # Compute TF-IDF
  tf_idf = {}
  for text in texts:
    cur_word_dict = {}
    cur_sent = text.split(' ')
    for word in cur_sent:
      if word not in tf_idf:
        tf_idf[word] = 0
      tf_idf[word] += 1. / len(cur_sent) * idf[word]
  return {
      "idf": idf,
      "tf_idf": tf_idf,
  }
  
class EfficientRandomGen(object):
  """A base class that generate multiple random numbers at the same time."""

  def reset_random_prob(self):
    """Generate many random numbers at the same time and cache them."""
    cache_len = 100000
    self.random_prob_cache = np.random.random(size=(cache_len,))
    self.random_prob_ptr = cache_len - 1

  def get_random_prob(self):
    """Get a random number."""
    value = self.random_prob_cache[self.random_prob_ptr]
    self.random_prob_ptr -= 1
    if self.random_prob_ptr == -1:
      self.reset_random_prob()
    return value

  def get_random_token(self):
    """Get a random token."""
    token = self.token_list[self.token_ptr]
    self.token_ptr -= 1
    if self.token_ptr == -1:
      self.reset_token_list()
    return token
    
class TfIdfWordRep(EfficientRandomGen):
  """TF-IDF Based Word Replacement."""

  def __init__(self, token_prob, data_stats):
    super(TfIdfWordRep, self).__init__()
    self.token_prob = token_prob
    self.data_stats = data_stats
    self.idf = data_stats["idf"]
    self.tf_idf = data_stats["tf_idf"]
    tf_idf_items = data_stats["tf_idf"].items()
    tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
    self.tf_idf_keys = []
    self.tf_idf_values = []
    for key, value in tf_idf_items:
      self.tf_idf_keys += [key]
      self.tf_idf_values += [value]
    self.normalized_tf_idf = np.array(self.tf_idf_values)
    self.normalized_tf_idf = (self.normalized_tf_idf.max()
                              - self.normalized_tf_idf)
    self.normalized_tf_idf = (self.normalized_tf_idf
                              / self.normalized_tf_idf.sum())
    self.reset_token_list()
    self.reset_random_prob()

  def get_replace_prob(self, all_words):
    """Compute the probability of replacing tokens in a sentence."""
    cur_tf_idf = collections.defaultdict(int)
    for word in all_words:
      cur_tf_idf[word] += 1. / len(all_words) * self.idf[word]
    replace_prob = []
    for word in all_words:
      replace_prob += [cur_tf_idf[word]]
    replace_prob = np.array(replace_prob)
    replace_prob = np.max(replace_prob) - replace_prob
    replace_prob = (replace_prob / replace_prob.sum() *
                    self.token_prob * len(all_words))
    return replace_prob

  def __call__(self, example):

    all_words = example.split(' ')

    replace_prob = self.get_replace_prob(all_words)
    all_words = self.replace_tokens(
        all_words,
        replace_prob[:len(all_words)]
        )

    return " ".join(all_words)

  def replace_tokens(self, word_list, replace_prob):
    """Replace tokens in a sentence."""
    for i in range(len(word_list)):
      if self.get_random_prob() < replace_prob[i]:
        word_list[i] = self.get_random_token()
    return word_list

  def reset_token_list(self):
    cache_len = len(self.tf_idf_keys)
    token_list_idx = np.random.choice(
        cache_len, (cache_len,), p=self.normalized_tf_idf)
    self.token_list = []
    for idx in token_list_idx:
      self.token_list += [self.tf_idf_keys[idx]]
    self.token_ptr = len(self.token_list) - 1
    print("sampled token list: {:s}".format(" ".join(self.token_list)))

  
def split_text(text,label,id_):
	text_list,label_list,id_list,poids=[],[],[],[]
	#text=classification.tokenizer.tokenize(text)
	
	decoup_1a = int(0.05*len(text))
	decoup_1b = int(0.95*len(text))
	decoup_2 = int(len(text)/2)
	decoup_3 = int(len(text)/3)
	decoup_4 = int(len(text)/4)
	decoup_5 = int(len(text)/5)
	'''
	#pour decoupage en un
	proportion_de_coupe=[0.15]
	for prop in proportion_de_coupe:
		debut=int(prop*len(text))
		fin=len(text)-debut
		text_list.append(text[debut:fin])
		label_list.append(label)
		id_list.append(id_)
		poids.append(1.0)
	'''
	'''
	#pour decoupage en n
	coupe=2
	decoupe = int(len(text)/coupe)
	for c in range(coupe):
		debut=int(c*decoupe)
		fin=int((c+1)*decoupe)
		text_list.append(text[debut:fin])
		label_list.append(label)
		id_list.append(id_)
		poids.append(1.0)
	'''
	
	#pour decoupage en word fenetre glissante
	word_list=text.split(' ')
	#word_list=text
	window_size=int(len(word_list)*0.9)
	marqueur=window_size
	while marqueur<len(word_list) :
		text_list.append(' '.join(word_list[marqueur-window_size:marqueur]))
		label_list.append(label)
		poids.append(1)
		id_list.append(id_)
		marqueur+=5
	

	'''
	#pour decoupage niveau un
	text_list  = [text[decoup_4:decoup_5]]
	label_list = [label]
	poids = [0.75]
	id_list=[id_]
	'''
	'''
	#pour decoupage niveau deux
	#text_list  = [text[decoup_1a:decoup_1b],text[:decoup_2],text[decoup_2:]]
	text_list  = [text[:decoup_2],text[decoup_2:]]
	label_list = [label,label]
	poids = [0.5,0.5]
	id_list=[id_,id_]
	'''
	
	#pour decoupage niveau trois
	text_list  = text_list+[text[decoup_1a:decoup_1b],text[:decoup_2],text[decoup_2:],text[:decoup_3],text[decoup_3:2*decoup_3],text[2*decoup_3:]]
	label_list = label_list+[label,label,label,label,label,label]
	poids =poids+ [0.9,0.5,0.5,0.3,0.3,0.3]
	id_list=id_list+[id_,id_,id_,id_,id_,id_]
	
	'''
	#pour decoupage niveau quatre
	text_list  = [text[decoup_1a:decoup_1b],text[:decoup_2],text[decoup_2:],text[:decoup_3],text[decoup_3:2*decoup_3],text[2*decoup_3:],text[decoup_4:],text[decoup_4:decoup_4*2],text[decoup_4*2:decoup_4*3],text[decoup_4*3:]]
	label_list = [label,label,label,label,label,label,label,label,label,label]
	poids = [0.75,0.5,0.5,0.3,0.3,0.3,0.25,0.25,0.25,0.25]
	id_list=[id_,id_,id_,id_,id_,id_,id_,id_,id_,id_]
	'''
	'''
	#pour decoupage niveau cinq
	text_list  = [text[decoup_1a:decoup_1b],text[:decoup_2],text[decoup_2:],text[:decoup_3],text[decoup_3:2*decoup_3],text[2*decoup_3:],text[decoup_4:],text[decoup_4:decoup_4*2],text[decoup_4*2:decoup_4*3],text[decoup_4*3:],text[:decoup_5],text[decoup_5:decoup_5*2],text[decoup_5*2:decoup_5*3],text[decoup_5*3:decoup_5*4],text[decoup_5*4:]]
	label_list = [label,label,label,label,label,label,label,label,label,label,label,label,label,label,label]
	poids = [0.75,0.5,0.5,0.3,0.3,0.3,0.25,0.25,0.25,0.25,0.2,0.2,0.2,0.2,0.2]
	id_list=[id_,id_,id_,id_,id_,id_,id_,id_,id_,id_,id_,id_,id_,id_,id_]
	'''

	return text_list,label_list,id_list,poids

def merge_text(texta,textb,label,id_):
	text_list,label_list,id_list,poids=[],[],[],[]
	tokena=classification.tokenizer.tokenize(texta)
	tokenb=classification.tokenizer.tokenize(texta)
	
	proportion_de_coupe =random.uniform(0, 1)
	idx_coupe_a = int(proportion_de_coupe*len(tokena))
	idx_coupe_b = int(proportion_de_coupe*len(tokenb))
	augmented_text=''.join(tokena[:idx_coupe_a])+''.join(tokena[idx_coupe_b:])
	
	text_list.append(augmented_text)
	label_list.append(label)
	id_list.append(id_)
	poids.append(1.0)

	return text_list,label_list,id_list,poids

def eda_text(text,label,id_):
	text_list,label_list,id_list=[],[],[]
	
	#pour decoupage en word
	word_list_1=text.split(' ')
	#inversion de deux mot
	idx_1 = random.randint(0,len(word_list_1)-1) 
	idx_2 = random.randint(0,len(word_list_1)-1) 
	word_list_1[idx_1],word_list_1[idx_2] = word_list_1[idx_2],word_list_1[idx_1]
	text_list = [' '.join(word_list_1)]
	label_list= [label]
	id_list   = [id_]
	#suppression d'un mot mot
	word_list_2=text.split(' ')
	idx_3 = random.randint(0,len(word_list_2)-1) 
	del word_list_2[idx_1]
	text_list.append(' '.join(word_list_2))
	label_list.append(label)
	id_list.append(id_)
	#Synonym Replacement
	word_list_3=text.split(' ')
	idx_4 = random.randint(0,len(word_list_3)-1) 
	if word_list_3[idx_4] not in stoplist:
		if len(wordnet.synsets(word_list_3[idx_4]))>0:
			idx_synonym=random.randint(0,len(wordnet.synsets(word_list_3[idx_4]))-1)
			synonym = wordnet.synsets(word_list_3[idx_4])[idx_synonym].lemma_names()[0]
			if synonym!=word_list_3[idx_4]:
				word_list_3[idx_4]=synonym
				text_list.append(' '.join(word_list_2))
				label_list.append(label)
				id_list.append(id_)
	#Random Insertion (RI)
	word_list_4=text.split(' ')
	idx_5 = random.randint(0,len(word_list_4)-1) 
	idx_6 = random.randint(0,len(word_list_4)-1) 
	if word_list_4[idx_5] not in stoplist:
		if len(wordnet.synsets(word_list_4[idx_5]))>0:
			idx_synonym=random.randint(0,len(wordnet.synsets(word_list_4[idx_5]))-1)
			synonym = wordnet.synsets(word_list_4[idx_5])[idx_synonym].lemma_names()[0]
			if synonym!=word_list_4[idx_5]:
				word_list_4.insert(idx_6, synonym)
				text_list.append(' '.join(word_list_2))
				label_list.append(label)
				id_list.append(id_)
	return text_list,label_list,id_list

	
def inversion_text(text,label):
	text_list,label_list=[],[]
	text_list  = [text[int(len(text)/2):]+text[:int(len(text)/2)]]
	label_list = [label]
	return text_list,label_list
	
def extract_id_augmentation(df_augment,idx):
	for id_ in idx:
		df_augment.loc[df_augment['id']==id_]
		frames = [train_origin.iloc[idx_train],]
		train = pd.concat(frames)
		
def prediction_inference_augmentation(predictions,df_test,idx_test):
	label_predits=[]
	df_prediction = pd.DataFrame(predictions,index=df_test.index)
	df=df_test.join(df_prediction)
	for idx in idx_test:
		df_idx=df[df['id'].isin([idx])]
		if 'poids' in df_idx.columns:
			poids=np.array(df_idx['poids'].values)
			del df_idx['poids']
		del df_idx['id']
		del df_idx['text']
		del df_idx['labels']
		np_pred=df_idx.to_numpy()

		#somme pondérée
		somme=[0]*len(np_pred[0])#pour recup nombre de classe!
		for i,vector in enumerate(np_pred):

				for j,proba in enumerate(vector):
					#somme[j]=somme[j]+proba*poids[i]
					somme[j]=somme[j]+proba
		
		label=np.argmax(somme,axis=0)
		label_predits.append(label)

	return label_predits
