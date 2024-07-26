"""
Adapted from comfyui CLIP code.
https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd1_clip.py
"""

import os

from transformers import T5Tokenizer, T5EncoderModel, T5Config, modeling_utils
import torch
import traceback
import zipfile
from comfy import model_management

from comfy.sd1_clip import parse_parentheses, token_weights, escape_important, unescape_important, safe_load_embed_zip, expand_directory_list, load_embed

class T5v11Model(torch.nn.Module):
    def __init__(self, textmodel_ver="xxl", textmodel_json_config=None, textmodel_path=None, device="cpu", max_length=120, freeze=True, dtype=None):
        super().__init__()

        self.num_layers = 24
        self.max_length = max_length
        self.bnb = False

        if textmodel_path is not None:
            model_args = {}
            model_args["low_cpu_mem_usage"] = True # Don't take 2x system ram on cpu
            if dtype == "bnb8bit":
                self.bnb = True
                model_args["load_in_8bit"] = True
            elif dtype == "bnb4bit":
                self.bnb = True
                model_args["load_in_4bit"] = True
            else:
                if dtype: model_args["torch_dtype"] = dtype
                self.bnb = False
            # second GPU offload hack part 2
            if device.startswith("cuda"):
                model_args["device_map"] = device
            print(f"Loading T5 from '{textmodel_path}'")
            self.transformer = T5EncoderModel.from_pretrained(textmodel_path, **model_args)
        else:
            if textmodel_json_config is None:
                textmodel_json_config = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    f"t5v11-{textmodel_ver}_config.json"
                )
            config = T5Config.from_json_file(textmodel_json_config)
            self.num_layers = config.num_hidden_layers
            with modeling_utils.no_init_weights():
                self.transformer = T5EncoderModel(config)

        if freeze:
            self.freeze()
        self.empty_tokens = [[0] * self.max_length] # <pad> token

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, tokens):
        device = self.transformer.get_input_embeddings().weight.device
        tokens = torch.LongTensor(tokens).to(device)
        attention_mask = torch.zeros_like(tokens)
        max_token = 1 # </s> token
        for x in range(attention_mask.shape[0]):
            for y in range(attention_mask.shape[1]):
                attention_mask[x, y] = 1
                if tokens[x, y] == max_token:
                    break

        outputs = self.transformer(input_ids=tokens, attention_mask=attention_mask)

        z = outputs['last_hidden_state']
        z.detach().cpu().float()
        return z

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=False)

    def to(self, *args, **kwargs):
        """BNB complains if you try to change the device or dtype"""
        if self.bnb:
            print("Thanks to BitsAndBytes, T5 becomes an immovable rock.", args, kwargs)
        else:
            self.transformer.to(*args, **kwargs)

    def encode_token_weights(self, token_weight_pairs, return_padded=False):
        to_encode = list(self.empty_tokens)
        for x in token_weight_pairs:
            tokens = list(map(lambda a: a[0], x))
            to_encode.append(tokens)

        out = self.encode(to_encode)
        z_empty = out[0:1]

        output = []
        for k in range(1, out.shape[0]):
            z = out[k:k+1]
            for i in range(len(z)):
                for j in range(len(z[i])):
                    weight = token_weight_pairs[k - 1][j][1]
                    z[i][j] = (z[i][j] - z_empty[0][j]) * weight + z_empty[0][j]
            output.append(z)

        if (len(output) == 0):
            return z_empty.cpu()

        out = torch.cat(output, dim=-2)
        if not return_padded:
            # Count number of tokens that aren't <pad>, then use that number as an index.
            keep_index = sum([sum([1 for y in x if y[0] != 0]) for x in token_weight_pairs])
            out = out[:, :keep_index, :]
        return out


class T5v11Tokenizer:
    """
    This is largely just based on the ComfyUI CLIP code.
    """
    def __init__(self, tokenizer_path=None, max_length=120, embedding_directory=None, embedding_size=4096, embedding_key='t5'):
        if tokenizer_path is None:
            tokenizer_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "t5_tokenizer")
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
        self.max_length = max_length
        self.max_tokens_per_section = self.max_length - 1 # </s> but no <BOS>

        self.pad_token = self.tokenizer("<pad>", add_special_tokens=False)["input_ids"][0]
        self.end_token = self.tokenizer("</s>", add_special_tokens=False)["input_ids"][0]
        vocab = self.tokenizer.get_vocab()
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.embedding_directory = embedding_directory
        self.max_word_length = 8 # haven't verified this
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def _try_get_embedding(self, embedding_name:str):
        '''
        Takes a potential embedding name and tries to retrieve it.
        Returns a Tuple consisting of the embedding and any leftover string, embedding can be None.
        '''
        embed = load_embed(embedding_name, self.embedding_directory, self.embedding_size, self.embedding_key)
        if embed is None:
            stripped = embedding_name.strip(',')
            if len(stripped) < len(embedding_name):
                embed = load_embed(stripped, self.embedding_directory, self.embedding_size, self.embedding_key)
                return (embed, embedding_name[len(stripped):])
        return (embed, "")

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        '''
        Takes a prompt and converts it to a list of (token, weight, word id) elements.
        Tokens can both be integer tokens and pre computed T5 tensors.
        Word id values are unique per word and embedding, where the id 0 is reserved for non word tokens.
        Returned list has the dimensions NxM where M is the input size of T5
        '''
        pad_token = self.pad_token
        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        #tokenize words
        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                #if we find an embedding, deal with the embedding
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                    #if we accidentally have leftover text, continue parsing using leftover, else move on to next word
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                #parse word
                tokens.append([(t, weight) for t in self.tokenizer(word, add_special_tokens=False)["input_ids"]])

        #reshape token array to T5 input size
        batched_tokens = []
        batch = []
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            #determine if we're going to try and keep the tokens in a single batch
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    #break word in two and add end token
                    if is_large:
                        batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                        batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    #add end token and pad
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        batch.extend([(self.pad_token, 1.0, 0)] * (remaining_length))
                    #start new batch
                    batch = []
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t,w,i+1) for t,w in t_group])
                    t_group = []

        # fill last batch
        batch.extend([(self.end_token, 1.0, 0)] + [(self.pad_token, 1.0, 0)] * (self.max_length - len(batch) - 1))
        # instead of filling, just add EOS (DEBUG)
        # batch.extend([(self.end_token, 1.0, 0)])

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]
        return batched_tokens

    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))
