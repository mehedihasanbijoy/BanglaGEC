import warnings
warnings.filterwarnings('ignore')

import torch
from torch.nn.utils.rnn import pad_sequence

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------