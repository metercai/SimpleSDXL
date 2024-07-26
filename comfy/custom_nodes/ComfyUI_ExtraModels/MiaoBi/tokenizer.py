import os
from transformers import AutoTokenizer
from comfy.sd1_clip import SDTokenizer

class MiaoBiTokenizer(SDTokenizer):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		tokenizer_path = os.path.join(
			os.path.dirname(os.path.realpath(__file__)),
			f"tokenizer"
		)
		# remote code ok, see `clip_tokenizer_roberta.py`, no ckpt vocab
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

		empty = self.tokenizer('')["input_ids"]
		if self.tokens_start:
			self.start_token = empty[0]
			self.end_token = empty[1]
		else:
			self.start_token = None
			self.end_token = empty[0]

		vocab = self.tokenizer.get_vocab()
		self.inv_vocab = {v: k for k, v in vocab.items()}