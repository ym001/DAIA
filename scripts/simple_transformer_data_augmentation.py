import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

import sys
####################
#importation des modules perso
path = "../module"
sys.path.append(path)
from Data import Data
from classification_model import ClassificationModel
from Data_augmentation import *

######################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.model_selection import train_test_split,KFold
from sklearn.manifold import TSNE

from tqdm import tqdm
import gc
import json
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet

import random
import math
import collections
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(0)
k_fold					= 4
text_augmentation		= True
inference_augmentation	= True
split_data				= True
merge					= False
uda						= False
eda						= False
back_translation		= False
word_level				= False
text_generator			= False
inversion_data			= False

PATH					= '../models/'
ID_COLUMN				= 'id'
POIDS_COLUMN			= 'poids'
DATA_COLUMN				= 'text'
LABEL_COLUMN			= 'labels'
EPOCH					=4
NB_TEXT					=5000

####################
donnees=Data()
label_list=donnees.list_label
print(label_list)
donnees.label_to_int()
train_origin=donnees.get_train()

# Create a ClassificationModel
#model_name='bert'; model_pretrained='bert-base-uncased'
model_name='roberta'; model_pretrained='roberta-base'
#model_name='clinicalbert'; model_pretrained='/home/mercadier/model/clinicalbert/model/pretraining/'
#model_name='scibert'; model_pretrained='/home/mercadier/model/scibert/scibert_scivocab_uncased/'
#model_name='xlnet'; model_pretrained='xlnet-base-cased'
#model_name='albert'; model_pretrained='albert-base-v1'
#model_name='distilbert'; model_pretrained='distilbert-base-uncased'
classification = ClassificationModel(model_name, model_pretrained, num_labels=len(label_list),args={'learning_rate':1e-5, 'num_train_epochs': EPOCH, 'reprocess_input_data': True, 'overwrite_output_dir': True})
classification.model.save_pretrained(PATH)

print('Augmentation')
if text_augmentation and word_level:
	stoplist = stopwords.words('english')
	list_text_augment=[]
	for text in data['text_original_en'][:NB_TEXT]:
		#list_text_augment.append(word_shuffle(text))
		list_text_augment.append(word_synonym(text))
	train_augmentation = pd.DataFrame({DATA_COLUMN:list_text_augment , LABEL_COLUMN:data['label'][:NB_TEXT]})

if text_augmentation and back_translation:
	list_text_back=[]
	for text in data['text_paraphrase_en']:
		list_text_back.append(text)
	train_augmentation = pd.DataFrame({DATA_COLUMN:list_text_back , LABEL_COLUMN:data['label'][:NB_TEXT], ID_COLUMN:id_origin})

if text_augmentation and text_generator :
	aug_text_gen=[]
	aug_label=[]
	aug_id=[]
	aug_poids=[]
	for text_init,list_gen,lab,id_init in zip(data['text_original'],data['text_generes'],data['label'],id_origin):
		for text in list_gen:
			aug_text_gen.append(text_init[:int(len(text_init)/2)]+text)
			aug_label.append(lab)
			aug_id.append(id_init)
			aug_poids.append([1.0])
	train_augmentation = pd.DataFrame({DATA_COLUMN:aug_text_gen , LABEL_COLUMN:aug_label, ID_COLUMN:aug_id,POIDS_COLUMN:aug_poids})
	
if text_augmentation and split_data :
	decoupage_text_augment=[]
	decoupage_label_augment=[]
	decoupage_id=[]
	decoupage_poids=[]
	#cpt_id=0
	for id_idx,text,label in tqdm(zip(train_origin['id'],train_origin['text'],train_origin['labels'])):
		text_list,label_list,id_list,poids_list=split_text(text,label,id_idx)
		decoupage_text_augment  = decoupage_text_augment+text_list
		decoupage_label_augment = decoupage_label_augment+label_list
		decoupage_id = decoupage_id+id_list
		decoupage_poids = decoupage_poids+poids_list
		#cpt_id+=1
	train_augmentation = pd.DataFrame({ID_COLUMN:decoupage_id ,POIDS_COLUMN:decoupage_poids ,DATA_COLUMN:decoupage_text_augment , LABEL_COLUMN:decoupage_label_augment})
	print('len(train_augmentation)')
	print(len(train_augmentation))

if text_augmentation and merge :
	merge_text_augment=[]
	merge_label_augment=[]
	merge_id=[]
	merge_poids=[]
	#cpt_id=0
	text_by_classe={}
	for i in range(len(label_list)):
		text_by_classe[str(i)]=[]
	for id_idx,text,label in zip(train_origin['id'],train_origin['text'],train_origin['labels']):
		text_by_classe[str(label)].append([id_idx,text,label])
	for key in text_by_classe:
		for i,text_de_la_classe in enumerate(text_by_classe[key]):
			if i<len(text_by_classe[key])-3:
				text_list,label_list,id_list,poids_list=merge_text(text_de_la_classe[1],text_by_classe[key][i+1][1],text_de_la_classe[2],text_de_la_classe[0])
				merge_text_augment  = merge_text_augment+text_list
				merge_label_augment = merge_label_augment+label_list
				merge_id            = merge_id+id_list
				merge_poids         = merge_poids+poids_list
				text_list,label_list,id_list,poids_list=merge_text(text_de_la_classe[1],text_by_classe[key][i+2][1],text_de_la_classe[2],text_de_la_classe[0])
				merge_text_augment  = merge_text_augment+text_list
				merge_label_augment = merge_label_augment+label_list
				merge_id            = merge_id+id_list
				merge_poids         = merge_poids+poids_list
				text_list,label_list,id_list,poids_list=merge_text(text_de_la_classe[1],text_by_classe[key][i+3][1],text_de_la_classe[2],text_de_la_classe[0])
				merge_text_augment  = merge_text_augment+text_list
				merge_label_augment = merge_label_augment+label_list
				merge_id            = merge_id+id_list
				merge_poids         = merge_poids+poids_list
	train_augmentation = pd.DataFrame({ID_COLUMN:merge_id ,POIDS_COLUMN:merge_poids ,DATA_COLUMN:merge_text_augment , LABEL_COLUMN:merge_label_augment})
	print('len(train_augmentation) : {}'.format(len(train_augmentation)))

if text_augmentation and eda :
	decoupage_text_augment=[]
	decoupage_label_augment=[]
	decoupage_id=[]
	stoplist = stopwords.words('english')

	for id_idx,text,label in zip(train_origin['id'],train_origin['text'],train_origin['labels']):
		text_list,label_list,id_list=eda_text(text,label,id_idx)
		decoupage_text_augment  = decoupage_text_augment+text_list
		decoupage_label_augment = decoupage_label_augment+label_list
		decoupage_id = decoupage_id+id_list
	train_augmentation = pd.DataFrame({ID_COLUMN:decoupage_id ,DATA_COLUMN:decoupage_text_augment , LABEL_COLUMN:decoupage_label_augment})
	print('len(train_augmentation)')
	print(len(train_augmentation))

if text_augmentation and uda :
	data_stats=get_data_stats(train_origin['text'].values)
	token_prob=0.9
	op = TfIdfWordRep(token_prob, data_stats)
	text_augment=[]
	label_augment=[]
	id_augment=[]
	for id_idx,text,label in zip(train_origin['id'],train_origin['text'],train_origin['labels']):
		text_aug=op(text)
		text_augment.append(text_aug)
		label_augment.append(label)
		id_augment.append(id_idx)
	train_augmentation = pd.DataFrame({ID_COLUMN:id_augment ,DATA_COLUMN:text_augment , LABEL_COLUMN:label_augment})
	print('len(train_augmentation)')
	print(len(train_augmentation))
	
if text_augmentation and inversion_data :
	decoupage_text_augment=[]
	decoupage_label_augment=[]
	for text,label in zip(train_origin['text'],train_origin['labels']):
		text_list,label_list=split_text(text,label)
		decoupage_text_augment  = decoupage_text_augment+text_list
		decoupage_label_augment = decoupage_label_augment+label_list
	train_augmentation = pd.DataFrame({DATA_COLUMN:decoupage_text_augment , LABEL_COLUMN:decoupage_label_augment})


print('fin Augmentation')
##############
#validation croisée
xfold	= np.zeros(len(train_origin))
skf 	= KFold(n_splits=k_fold)
skf.get_n_splits(xfold)

acc_cross_validation=[]
acc_cross_validation_augmentation=[]
cpt_passe=0
for idx_train, idx_test in skf.split(xfold):
	print('Passe : {}'.format(cpt_passe))
	cpt_passe+=1
	#inversion of the cross-validation indices.
	idx_train, idx_test=idx_test,idx_train
	#reset du modèle
	classification.model = classification.model_class.from_pretrained(PATH)
	if text_augmentation:
		frames = [train_origin.iloc[idx_train],train_augmentation[train_augmentation['id'].isin(idx_train)]]
		train = pd.concat(frames)
	else:
		train=train_origin.iloc[idx_train]

	train_features=classification.extraction_features(train)
	
	# Train the model
	classification.train_model(train_features)
	# Evaluate the model
	if inference_augmentation:
		frames_test 							= [train_origin[train_origin['id'].isin(idx_test)],train_augmentation[train_augmentation['id'].isin(idx_test)]]
		test 									= pd.concat(frames_test)
		test_features_augmentation				= classification.extraction_features(test)
		test_features_only_augmentation			= classification.extraction_features(train_augmentation[train_augmentation['id'].isin(idx_test)])
		test_features							= classification.extraction_features(train_origin.iloc[idx_test])
		predictions,embeddings 					= classification.predict(test_features)
		predictions_only_augmentation,embeddings_only_augmentation = classification.predict(test_features_only_augmentation)
		predictions_augmentation,embeddings_augmentation = classification.predict(test_features_augmentation)
	else:
		test_features							= classification.extraction_features(train_origin.iloc[idx_test])
		predictions,embeddings 					= classification.predict(test_features)
	if inference_augmentation:
		label_predits							= np.argmax(predictions, axis=1)
		label_predits_augmented					= np.argmax(predictions_augmentation, axis=1)
		label_predits_only_augmented			= np.argmax(predictions_only_augmentation, axis=1)
		label_predits_inference_augmentation	=prediction_inference_augmentation(predictions_augmentation,test,idx_test)
		
		acc=sklearn.metrics.accuracy_score(train_origin.iloc[idx_test]['labels'].values.tolist(),label_predits)
		acc_cross_validation.append(acc)
		print("da {} {} acc :{} = {} ep:{}  max_seq_length : {}".format(model_name,len(train_origin),np.mean(acc_cross_validation),acc_cross_validation,EPOCH,classification.max_seq_length))
		acc_augmentation=sklearn.metrics.accuracy_score(train_origin.iloc[idx_test]['labels'].values.tolist(),label_predits_inference_augmentation)
		acc_cross_validation_augmentation.append(acc_augmentation)
		print("daia {} {} acc :{} = {} ep:{}  max_seq_length : {}".format(model_name,len(train_origin),np.mean(acc_cross_validation_augmentation),acc_cross_validation_augmentation,EPOCH,classification.max_seq_length))
	
	else:
		label_predits=np.argmax(predictions, axis=1)
		acc=sklearn.metrics.accuracy_score(train_origin.iloc[idx_test]['labels'].values.tolist(),label_predits)
		acc_cross_validation.append(acc)
		print("{} {} acc :{} = {} ep:{}  max_seq_length : {}".format(model_name,len(train_origin),np.mean(acc_cross_validation),acc_cross_validation,EPOCH,classification.max_seq_length))

print('End of script.')
