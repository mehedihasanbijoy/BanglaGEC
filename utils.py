import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm 
import pandas as pd
from sklearn.model_selection import train_test_split





# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def punctuations_preprocessing(sent, punc_list=[',', '।', '.']):
    sent = [_ for _ in sent]

    for idx in range(len(sent)):
        if idx != 0 and sent[idx] in punc_list and sent[idx-1] == ' ':
            sent[idx-1] = ''
    
    return ''.join(sent)
    
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def tokenizeInstances(tokenizer, df):

	train_sources_encodings = []
	train_mask_encodings = []
	train_targets_encodings = []

	for i in tqdm(range(len(df))):
		correct = df['Correct'][i]
		erroneous = df['Erroneous'][i]
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

	return (train_sources_encodings, train_mask_encodings, train_targets_encodings)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def trainValidationTestSplit(df, valid_size, test_size):
	etypes = list(set(df['ErrorType']))

	train_df = pd.DataFrame()
	valid_df = pd.DataFrame()
	test_df = pd.DataFrame()

	for etype in etypes:
		etype_df = df.loc[df['ErrorType'] == etype]
		train, test = train_test_split(etype_df, test_size=test_size)
		train, valid = train_test_split(train, test_size=valid_size)

		train_df = pd.concat([train_df, train])
		valid_df = pd.concat([valid_df, valid])
		test_df = pd.concat([test_df, test])

	train_df = train_df.sample(frac=1).reset_index(drop=True)
	valid_df = valid_df.sample(frac=1).reset_index(drop=True)
	test_df = test_df.sample(frac=1).reset_index(drop=True)

	return (train_df, valid_df, test_df)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------