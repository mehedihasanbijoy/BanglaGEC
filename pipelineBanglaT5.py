import os
import random 
from tqdm import tqdm 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AdamW
import datasets

from utils import tokenizeInstances
from customDataset import LoadDataset, collate_fn

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
	

	train_df = pd.read_csv('/home/uiu/nlp/GEC/BanglaGEC/corpus/train.csv')
	valid_df = pd.read_csv('/home/uiu/nlp/GEC/BanglaGEC/corpus/valid.csv')
	test_df = pd.read_csv('/home/uiu/nlp/GEC/BanglaGEC/corpus/test.csv')

	print(f"#no. of training instances: {len(train_df)}")
	print(f"#no. of validation instances: {len(valid_df)}")
	print(f"#no. of test instances: {len(test_df)}")
	print(f"Total: {len(train_df) + len(valid_df) + len(test_df)}")

	print("train and test df are loaded")

	# ---------------------------------------------------------------

	tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_banglaparaphrase", use_fast=False)

	# ---------------------------------------------------------------

	print("training instances are being tokenized")
	train_sources_encodings, train_mask_encodings, train_targets_encodings = tokenizeInstances(tokenizer, train_df)
	
	# ---------------------------------------------------------------

	print("validation instances are being tokenized")
	valid_sources_encodings, valid_mask_encodings, valid_targets_encodings = tokenizeInstances(tokenizer, valid_df)

	# ---------------------------------------------------------------

	print("test instances are being tokenized")
	test_sources_encodings, test_mask_encodings, test_targets_encodings = tokenizeInstances(tokenizer, test_df)

	# ---------------------------------------------------------------

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# ---------------------------------------------------------------

	train_dataset = LoadDataset(train_sources_encodings, train_mask_encodings, train_targets_encodings)
	valid_dataset = LoadDataset(valid_sources_encodings, valid_mask_encodings, valid_targets_encodings)
	test_dataset = LoadDataset(test_sources_encodings, test_mask_encodings, test_targets_encodings)

	# ---------------------------------------------------------------

	train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=16, collate_fn=collate_fn, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn, shuffle=False)
	print("training, validation and test dataloaders are in action with collate fn")

	# ---------------------------------------------------------------

	model_checkpoint = 'csebuetnlp/banglat5_banglaparaphrase'  # 990M
	# model_checkpoint = 't5-small'  # 242M
	# model_checkpoint = 'Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU'  # MarianMT 298M
	model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
	model.to(device)
	print("model is in gpu now")

	# ---------------------------------------------------------------

	optim = AdamW(model.parameters(), lr=5e-5)

	N_EPOCHS = 200
	epoch = 0

	loss = 10e9

	# ---------------------------------------------------------------
	print("transfering the knowledge")

	PATH = '/home/uiu/nlp/GEC/HFPipeline/checkpoints/saved_model_BPBanglaT5.pth'

	if os.path.exists(PATH):
		checkpoint = torch.load(PATH)
		model.load_state_dict(checkpoint['model_state_dict'])

	print("knowledge transfered")

	# ---------------------------------------------------------------

	PATH = '/home/uiu/nlp/GEC/HFPipeline/checkpoints/GECBanglaT5.pth' # banglaT5 76.96%
	# PATH = '/home/uiu/nlp/GEC/HFPipeline/checkpoints/saved_model_gecV4T5Small.pth' # T5Small 71.44%

	if os.path.exists(PATH):
		checkpoint = torch.load(PATH)
		model.load_state_dict(checkpoint['model_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']  

	print("incorporated model checkpoint")
	# ---------------------------------------------------------------

	print("training has started")
	for epoch in range(epoch, N_EPOCHS):
		print(f"Epoch = {epoch}")
		epoch_loss = 0
		model.train()

		for (input_ids, attention_mask, target_ids) in tqdm(train_loader):
			input_ids = input_ids.to(device)
			attention_mask = attention_mask.to(device)
			target_ids = target_ids.to(device)

			optim.zero_grad()
			predictions = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
			loss = predictions[0]
			loss.backward()
			epoch_loss += loss.item()
			optim.step()
		
		epoch_loss = epoch_loss/len(train_loader)
		print(f"Loss = {epoch_loss}")

		if epoch_loss < loss:
			loss = epoch_loss
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'loss': loss,
			}, PATH)
			print(f"{'-'*20}\nModel Saved at {PATH}\n{'-'*20}\n")
	# Training Loop Ends Here

	# ---------------------------------------------------------------

	PATH = '/home/uiu/nlp/GEC/HFPipeline/checkpoints/GECBanglaT5.pth' # banglaT5 76.96%
	# PATH = '/home/uiu/nlp/GEC/HFPipeline/checkpoints/saved_model_gecV4T5Small.pth' # T5Small 71.44%

	if os.path.exists(PATH):
		checkpoint = torch.load(PATH)
		model.load_state_dict(checkpoint['model_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']  

	print("incorporated model checkpoint")

	# ---------------------------------------------------------------

	print("evaluation has started")
	model.eval()
	all_preds = []

	true_corrections = []
	pred_outputs = []

	for (input_ids, attention_mask, target_ids) in tqdm(test_loader):
		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)
		target_ids = target_ids.to(device)

		predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask)
		# predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
		# print(predictions.shape, target_ids.shape)

		trg_text = [tokenizer.decode(token, skip_special_tokens=True) for token in target_ids]
		prd_text = [tokenizer.decode(token, skip_special_tokens=True) for token in predictions]
		# print(prd_text)
		# print(trg_text)

		true_corrections += trg_text
		pred_outputs += prd_text

		# all_preds.extend([x == y for x, y in zip(prd_text, trg_text)])
		
		# predictions = predictions[1]
		# print(torch.argmax(predictions, dim= -1).shape) 
	# print(f"Accuracy: {sum(all_preds) / len(all_preds) * 100 : .2f}%")

	acc = accuracy_score(y_true=true_corrections, y_pred=pred_outputs)
	pr = precision_score(y_true=true_corrections, y_pred=pred_outputs, average='micro')
	re = recall_score(y_true=true_corrections, y_pred=pred_outputs, average='micro')
	# f1 = f1_score(y_true=true_corrections, y_pred=pred_outputs, average='micro')
	f1 = fbeta_score(y_true=true_corrections, y_pred=pred_outputs, average='micro', beta=1)
	f05 = fbeta_score(y_true=true_corrections, y_pred=pred_outputs, average='micro', beta=0.5)

	print(f"Accuracy Score = {acc*100:.2f}%")
	print(f"Precision Score = {pr:.5f}")
	print(f"Recall Score = {re:.5f}")
	print(f"F1 Score = {f1:.5f}")
	print(f"F0.5 Score = {f05:.5f}")

	# Evaluation Loop Ends Here

	# ---------------------------------------------------------------

	# for _ in range(20):
	#     idx = random.randint(0, 10)
	#     src_text = test_df['Erroneous'][idx]
	#     trg_text = test_df['Correct'][idx]

	#     predicted = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to(device))
	#     prd_text = [tokenizer.decode(token, skip_special_tokens=True) for token in predicted][0]

	#     print(f"Err: {src_text}\nPrd: {prd_text}\nTrg: {trg_text}\n{prd_text == trg_text}\n")
	# prediction generation

	# ---------------------------------------------------------------


	preds = []
	refs = []
	refs4BERTscore = []

	for idx in tqdm(range(len(test_df))):
	# for idx in tqdm(range(1000)):
		# idx = random.randint(0, 1000)
		src_text = test_df['Erroneous'][idx]
		trg_text = test_df['Correct'][idx]

		predicted = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to(device))
		prd_text = [tokenizer.decode(token, skip_special_tokens=True) for token in predicted][0]
		# prd_text = ' '.join(([tokenizer.decode(token, skip_special_tokens=True) for token in predicted][0]).split()[1:])
		# prd_text = [tokenizer.decode(token, skip_special_tokens=True) for token in predicted][0]

		if idx % 100 == 0:
			print(f"Err: {src_text}\nPrd: {prd_text}\nTrg: {trg_text}\n{prd_text == trg_text}\n")

		preds.append(prd_text)
		refs.append([trg_text])
		refs4BERTscore.append(trg_text)
	# prediction generation

	sacrebleu = datasets.load_metric('sacrebleu')
	sacrebleuResults = sacrebleu.compute(predictions=preds, references=refs)
	print(f"sacrebleuResults = {round(sacrebleuResults['score'], 1)}")

	# rougescore = datasets.load_metric('rouge')
	# rougescoreResults = rougescore.compute(predictions=preds, references=refs4BERTscore)
	# print(f"rougeLscoreResults = {rougescoreResults['rougeL'].high.fmeasure*100}")
	# # print(f"{rougescoreResults['rougeL'].high.fmeasure*100}")

	bertscore = datasets.load_metric('bertscore')
	bertscoreResults = bertscore.compute(predictions=preds, references=refs4BERTscore, lang="bn")
	print(f"bertscoreResults = {sum(bertscoreResults['f1'])/len(bertscoreResults['f1'])*100:.2f}")
	# print(f"{sum(bertscoreResults['f1'])/len(bertscoreResults['f1'])*100:.2f}")









































