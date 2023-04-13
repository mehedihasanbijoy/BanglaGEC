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

import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":
	# ---------------------------------------------------------------

	# df = pd.read_csv('/home/uiu/nlp/GEC/corpus/gecV4.csv')
	# # df = df.iloc[:(len(df) // 10) * 8, :]
	# df = df.iloc[:200000, :]
	# print("df loaded")
	# print(f"Total no of instances: {len(df)}")

	# train_df, test_df = train_test_split(df, test_size=0.1)
	# train_df = train_df.reset_index(drop=True)
	# test_df = test_df.reset_index(drop=True)
	# train_df.to_csv('/home/uiu/nlp/GEC/corpus/train_df_gecV4.csv', index=False)
	# test_df.to_csv('/home/uiu/nlp/GEC/corpus/test_df_gecV4.csv', index=False)

	# ---------------------------------------------------------------

	train_df = pd.read_csv('/home/uiu/nlp/GEC/corpus/train_df_gecV4.csv')
	test_df = pd.read_csv('/home/uiu/nlp/GEC/corpus/test_df_gecV4.csv')
	print(f"Total no of training instances: {len(train_df)}")
	print(f"Total no of test instances: {len(test_df)}")
	print("train and test df are loaded")

	# ---------------------------------------------------------------

	tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5_banglaparaphrase", use_fast=False)

	# ---------------------------------------------------------------

	print("training instances are being tokenized")
	train_sources_encodings = []
	train_mask_encodings = []
	train_targets_encodings = []

	for i in tqdm(range(len(train_df))):
	    correct = train_df['Correct'][i]
	    erroneous = train_df['Erroneous'][i]
	    # print(correct) 
	    # print(erroneous)  
	    correct_encoding = tokenizer(correct)
	    erroneous_encoding = tokenizer(erroneous)
	    # print(correct_encoding)
	    # print(erroneous_encoding)
	    train_source_encoding = erroneous_encoding['input_ids']
	    train_mask_encoding = erroneous_encoding['attention_mask']
	    train_target_encoding = correct_encoding['input_ids']
	    # print(train_source_encoding)
	    # print(train_mask_encoding)
	    # print(train_target_encoding)
	    train_sources_encodings.append(train_source_encoding)
	    train_mask_encodings.append(train_mask_encoding)
	    train_targets_encodings.append(train_target_encoding)
	    # break
	# 
	
	# ---------------------------------------------------------------

	print("test instances are being tokenized")
	test_sources_encodings = []
	test_mask_encodings = []
	test_targets_encodings = []

	for i in tqdm(range(len(test_df))):
	    correct = test_df['Correct'][i]
	    erroneous = test_df['Erroneous'][i]
	    # print(correct) 
	    # print(erroneous)  
	    correct_encoding = tokenizer(correct)
	    erroneous_encoding = tokenizer(erroneous)
	    # print(correct_encoding)
	    # print(erroneous_encoding)
	    test_source_encoding = erroneous_encoding['input_ids']
	    test_mask_encoding = erroneous_encoding['attention_mask']
	    test_target_encoding = correct_encoding['input_ids']
	    # print(train_source_encoding)
	    # print(train_mask_encoding)
	    # print(train_target_encoding)
	    test_sources_encodings.append(test_source_encoding)
	    test_mask_encodings.append(test_mask_encoding)
	    test_targets_encodings.append(test_target_encoding)
	    # break
	# 

	# ---------------------------------------------------------------

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# ---------------------------------------------------------------

	class LoadDataset(torch.utils.data.Dataset):
	    def __init__(self, source_encodings, mask_encodings, targets_encodings):
	        self.source_encodings = source_encodings
	        self.mask_encodings = mask_encodings
	        self.targets_encodings = targets_encodings
	    
	    def __getitem__(self, idx):
	        input_ids = torch.tensor(self.source_encodings[idx]).squeeze()
	        attention_mask = torch.tensor(self.mask_encodings[idx]).squeeze()
	        target_ids = torch.tensor(self.targets_encodings[idx]).squeeze()
	        return input_ids, attention_mask, target_ids
	    
	    def __len__(self):
	        return len(self.source_encodings)
	
	# ---------------------------------------------------------------

	train_dataset = LoadDataset(train_sources_encodings, train_mask_encodings, train_targets_encodings)
	test_dataset = LoadDataset(test_sources_encodings, test_mask_encodings, test_targets_encodings)


	# ---------------------------------------------------------------

	def collate_fn(batch):
	    sources, masks, targets = [], [], []

	    for (_source, _mask, _target) in batch:
	        sources.append(_source)
	        masks.append(_mask)
	        targets.append(_target)

	    sources = pad_sequence(sources, batch_first=True, padding_value=0)
	    masks = pad_sequence(masks, batch_first=True, padding_value=0)
	    targets = pad_sequence(targets, batch_first=True, padding_value=0)

	    return sources, masks, targets

	# ---------------------------------------------------------------

	train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=collate_fn, shuffle=True)
	test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn, shuffle=False)
	print("training and test dataloaders are in action with collate fn")

	# ---------------------------------------------------------------

	# model_checkpoint = 'csebuetnlp/banglat5_banglaparaphrase'  # 990M
	# model_checkpoint = 't5-small'  # 242M
	model_checkpoint = 'Helsinki-NLP/opus-mt-NORTH_EU-NORTH_EU'  # MarianMT 298M
	model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
	model.to(device)
	print("model is in gpu now")

	# ---------------------------------------------------------------

	optim = AdamW(model.parameters(), lr=5e-5)

	N_EPOCHS = 50
	epoch = 0

	loss = 10e9

	# ---------------------------------------------------------------
	print("transfering the knowledge")

	PATH = '/home/uiu/nlp/GEC/HFPipeline/checkpoints/saved_model_BPMarianMT.pth'

	if os.path.exists(PATH):
	    checkpoint = torch.load(PATH)
	    model.load_state_dict(checkpoint['model_state_dict'])

	print("knowledge transfered")

	# ---------------------------------------------------------------


	# PATH = '/home/uiu/nlp/GEC/HFPipeline/checkpoints/saved_model_gecV4.pth' # banglaT5 76.96%
	# PATH = '/home/uiu/nlp/GEC/HFPipeline/checkpoints/saved_model_gecV4T5Small.pth' # T5Small 71.44%
	PATH = '/home/uiu/nlp/GEC/HFPipeline/checkpoints/saved_model_gecV4OpusMarianMT.pth' # MarianMT 93.07%

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

	# PATH = '/home/uiu/nlp/GEC/HFPipeline/checkpoints/GECBanglaT5.pth' # banglaT5 76.96%
	PATH = '/home/uiu/nlp/GEC/HFPipeline/checkpoints/saved_model_gecV4OpusMarianMT.pth' # 

	if os.path.exists(PATH):
	    checkpoint = torch.load(PATH)
	    model.load_state_dict(checkpoint['model_state_dict'])
	    epoch = checkpoint['epoch']
	    loss = checkpoint['loss']  

	print("incorporated model checkpoint")

	# ---------------------------------------------------------------


	# ---------------------------------------------------------------

	print("evaluation has started")
	model.eval()
	all_preds = []

	true_corrections = []
	pred_outputs = []

	for (input_ids, attention_mask, target_ids) in test_loader:
	    input_ids = input_ids.to(device)
	    attention_mask = attention_mask.to(device)
	    target_ids = target_ids.to(device)

	    predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask)
	    # predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
	    # print(predictions.shape, target_ids.shape)

	    trg_text = [tokenizer.decode(token, skip_special_tokens=True) for token in target_ids]
	    prd_text = [tokenizer.decode(token, skip_special_tokens=True).replace('<extra_id_-25912>', '')[1:] for token in predictions]
	    # prd_text = [' '.join(tokenizer.decode(token, skip_special_tokens=True).split()[1:]) for token in predictions]
	    # prd_text = [' '.join(tokenizer.decode(token, skip_special_tokens=True).split()[1:]) for token in predictions]
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

	# # ---------------------------------------------------------------

	# for _ in range(50):
	#     idx = random.randint(0, 1000)
	#     src_text = test_df['Erroneous'][idx]
	#     trg_text = test_df['Correct'][idx]

	#     predicted = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to(device))
	#     prd_text = ' '.join(([tokenizer.decode(token, skip_special_tokens=True) for token in predicted][0]).split()[1:])
	#     # prd_text = [tokenizer.decode(token, skip_special_tokens=True) for token in predicted][0]

	#     print(f"Err: {src_text}\nPrd: {prd_text}\nTrg: {trg_text}\n{prd_text == trg_text}\n")
	# # prediction generation

	# # ---------------------------------------------------------------

	# ---------------------------------------------------------------

	preds = []
	refs = []
	refs4BERTscore = []

	for idx in tqdm(range(len(test_df))):
	# for idx in range(1000):
	    # idx = random.randint(0, 1000)
	    src_text = test_df['Erroneous'][idx]
	    trg_text = test_df['Correct'][idx]

	    predicted = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to(device))
	    prd_text = ' '.join(([tokenizer.decode(token, skip_special_tokens=True) for token in predicted][0]).split()[1:])
	    # prd_text = [tokenizer.decode(token, skip_special_tokens=True) for token in predicted][0]

	    if idx % 1000 == 0:
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
	# ---------------------------------------------------------------








































