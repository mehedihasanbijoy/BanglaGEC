import warnings
warnings.filterwarnings('ignore')

from utils import trainValidationTestSplit
import argparse
import pandas as pd


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--CORPUS", help="Path of the Corpus", type=str, default="/home/uiu/nlp/GEC/corpus/gecV4.csv", 
		choices=[
			"/home/uiu/nlp/GEC/corpus/gecV4.csv"
		]
	)
	parser.add_argument("--VALID_SIZE", help="Validation DF Size", type=float, default=0.1, choices=[0.05, 0.1, 0.2])
	parser.add_argument("--TEST_SIZE", help="Test DF Size", type=float, default=.1, choices=[0.05, 0.1, 0.2])
	parser.add_argument("--DESTINATION", help="Path of the Destination", type=str, default="/home/uiu/nlp/GEC/BanglaGEC/corpus/")

	args = parser.parse_args()

	df = pd.read_csv(args.CORPUS)

	train_df, valid_df, test_df = trainValidationTestSplit(df, args.VALID_SIZE, args.TEST_SIZE)

	train_df.to_csv(args.DESTINATION + 'train.csv', index=False)
	valid_df.to_csv(args.DESTINATION + 'valid.csv', index=False)
	test_df.to_csv(args.DESTINATION + 'test.csv', index=False)

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def checkSplits():
	train_df = pd.read_csv('/home/uiu/nlp/GEC/BanglaGEC/corpus/train.csv')
	valid_df = pd.read_csv('/home/uiu/nlp/GEC/BanglaGEC/corpus/valid.csv')
	test_df = pd.read_csv('/home/uiu/nlp/GEC/BanglaGEC/corpus/test.csv')

	print(f"len(train_df) = {len(train_df)}")
	print(f"len(valid_df) = {len(valid_df)}")
	print(f"len(test_df) = {len(test_df)}")
	print(f"Total = {len(train_df) + len(valid_df) + len(test_df)}")

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

if __name__ == "__main__":
	main()
	checkSplits()

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------