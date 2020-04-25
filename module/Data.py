#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  data.py
#  
#  Copyright 2017 yves <yves.mercadier@ac-montpellier.fr>
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
import re
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

class Data:  
	
	def __init__(self,taille_par_nb_echantillon=True,pourcentage_de_la_base=False,nb_echantillon=5000): # constructeur
		self.pourcentage_de_la_base=pourcentage_de_la_base
		self.taille_par_nb_echantillon=taille_par_nb_echantillon
		self.nb_echantillon=nb_echantillon
		
		docs,labels  =self.lecture_du_jeu_de_20news()
		self.df  = pd.DataFrame({'text':docs,'labels':labels})
		
		self.reduction_train()
		self.df_train_doc=self.nettoyage_df(self.df_train_doc)
		self.calcul_longueur_sequence()
		self.list_label=self.list_label(self.df_train_label.values.tolist())
		self.construct_id()
		
	#####################
	#lecture du jeu de données news
	#####################
	def lecture_du_jeu_de_20news(self):
		categorie = ['sci.crypt', 'sci.electronics','sci.med', 'sci.space','rec.autos','rec.motorcycles','rec.sport.baseball','rec.sport.hockey','talk.politics.guns','talk.politics.mideast','talk.politics.misc','talk.religion.misc']
		twenty_train = fetch_20newsgroups(subset='train',categories=categorie, shuffle=True, random_state=42)
		doc=twenty_train.data
		label=[]
		for i in range(len(twenty_train.target)):
			label.append([categorie[twenty_train.target[i]]])
		return doc,label
		
		
	def reduction_train(self):
		print("taille des doc "+str(len(self.df)))
		print(self.df)
		if self.taille_par_nb_echantillon == True:
			if self.nb_echantillon < self.df.shape[0]:
				proportion = 1-self.nb_echantillon*1.0/self.df.shape[0]
				self.df_train_doc, self.df_test_doc, self.df_train_label, self.df_test_label = train_test_split(self.df['text'], self.df['labels'], test_size=proportion, random_state=42,stratify=self.df['labels'])
				print("taille des doc reduite "+str(len(self.df_train_doc)))

	def nettoyage_df(self,df):
		from nltk.corpus import stopwords
		stop_unicode = stopwords.words('english')
		#conversion du dictionnaire en str
		stop=[str(w) for w in stop_unicode]
		df = df.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))#stop word
		
		df = df.apply(lambda x: re.sub(re.compile('<.*?>'), '', x))#supprime balise html
		
		#remove contraction
		def _get_contractions(contraction_dict):
			contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
			return contraction_dict, contraction_re

		contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
		contractions, contractions_re = _get_contractions(contraction_dict)
		def replace_contractions(text):
			def replace(match):
				return contractions[match.group(0)]
			return contractions_re.sub(replace, text)
		df = df.apply(lambda x: replace_contractions(x))
		
		df = df.apply(lambda x: re.sub(re.compile('[^a-zA-z0-9\s]'), ' ', x))#supprime les caracteres speciaux
		
		df = df.apply(lambda x: x.lower())#passe en minuscule
		
		def clean_numbers(x):
			if bool(re.search(r'\d', x)):
				x = re.sub('[0-9]{5,}', '#####', x)
				x = re.sub('[0-9]{4}', '####', x)
				x = re.sub('[0-9]{3}', '###', x)
				x = re.sub('[0-9]{2}', '##', x)
			return x
		df = df.apply(lambda x: clean_numbers(x))

		df = df.apply(lambda x: re.sub("[ ]{2,}", " ", x))#enleve les espaces successsifs
		df = df.apply(lambda x: x.strip())#enleve les espaces debut et fin

		lemmatizer = WordNetLemmatizer()
		df = df.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
		print(df.head())
		return df
		
	def calcul_longueur_sequence(self):
		count_sequence={}
		max_longueur=0
		c=0
		somme=0
		for item in self.df_train_doc.iteritems():

			sequence=item[1].split(' ')
			longueur=len(sequence)
			somme+=longueur
			if longueur in count_sequence:
				count_sequence[longueur]+=1
				if max_longueur<longueur:max_longueur=longueur
			else:count_sequence[longueur]=1

		print("sequence la plus longue: {}".format(max_longueur))
		print("nb mot moyen par séquence: {}".format(somme/len(self.df_train_doc)))
		somme_sequence=0
		for key in sorted(count_sequence.keys()):
			if somme_sequence<len(self.df_train_doc)*0.8:
				somme_sequence+=count_sequence[key]
				self.longueur_sequence=key
				
		print("longueur sequence choisie: {}".format(self.longueur_sequence))
		
	def label_to_int(self):
		def label_int(label):
			if label in self.list_label:
				idx_label=self.list_label.index(label)
			return idx_label
		self.df_train_label = self.df_train_label.apply(lambda x: label_int(x))#converti les labels str en int
		
	def construct_id(self):
		return [id_idx for id_idx in range(len(self.df_train_label))]
		
	def construct_poids(self):
		return [1.0 for id_idx in range(len(self.df_train_label))]
		
	#pour bert tensorflow
	def get_train(self):
		DATA_COLUMN = 'text'
		LABEL_COLUMN = 'labels'
		ID_COLUMN = 'id'
		#rajouté pour daia somme pondérée!
		POIDS_COLUMN = 'poids'
		#
		df_train = pd.DataFrame({ID_COLUMN:self.construct_id() ,POIDS_COLUMN:self.construct_poids() ,DATA_COLUMN:self.df_train_doc , LABEL_COLUMN:self.df_train_label})
		return df_train

	def get_doc_train(self):
		df_train = pd.DataFrame({'text':self.df_train_doc})
		return df_train
		
	def list_label(self,label_jeu):
		label=[]
		for l in label_jeu:
			if l not in label:
				label.append(l)
		label.sort(reverse=False)
		return label
