import torch
import pickle
import numpy as np


# Load the dictionary
with open('../data/parliment_names/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# In the loaded model I've tokenized my firends names for privacy. 
# Here i replace the name_i token by 'Member i:' text
new_decoded_members={
    '丁': '] Member 1: ',
    '丂': '] Member 2: ',
    '七': '] Member 3: ',
    '丄': '] Member 4: ',
    '丅': '] Member 5: ',
    '丆': '] Member 6: ',
    '万': '] Member 7: ',
    '丈': '] Member 8: ',
    '三': '] Member 9: '
    }

member_tokens = list(new_decoded_members.keys())

# build a new decoder that replace the tokens by 'new_decoded_members' when decoding
itos_member=itos.copy()
for tup in itos_member.items():
    if tup[1] in new_decoded_members:
        itos_member[tup[0]]=new_decoded_members[tup[1]]
decode_member = lambda l: ''.join([itos_member[i] for i in l])




class AttentionVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.num_layers = len(model.transformer.h)
        self.num_heads = model.transformer.h[0].attn.n_head
        self.attention_data = {}
        self.hooks = []
        self.prompt_text = ""
        self.att_arr = None

    def _attention_hook(self, layer_idx):
        def hook(module, input, output):
            self.attention_data[layer_idx] = module.att_amps.detach()
        return hook

    def _register_hooks(self):
        self.hooks = []
        for idx in range(self.num_layers):
            block_to_hook = self.model.transformer.h[idx].attn
            hook = block_to_hook.register_forward_hook(self._attention_hook(idx))
            self.hooks.append(hook)

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def prompt(self, sample_text):
        self.attention_data = {}
        self._register_hooks()
        self.prompt_text = sample_text


        # Encode the sample text
        ids = encode(sample_text)
        x = torch.tensor([ids], dtype=torch.long).to(self.device)

        # Process the input through the model
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(x)

        self._remove_hooks()

        # Organize the collected attention data into a tensor
        T = next(iter(self.attention_data.values())).size(-1)  # Sequence length
        self.att_arr = torch.empty(self.num_layers, self.num_heads, T, T)
        for layer_idx in range(self.num_layers):
            if layer_idx in self.attention_data:
                self.att_arr[layer_idx] = self.attention_data[layer_idx]

    def visualize_specific_attention_html(self, layer_head_pairs=None, num_last_tokens=None, show_layer_head_info=False):
        text = self.prompt_text
        att_arr = self.att_arr
        text_and_line_color = '#808080'

        if layer_head_pairs is None or not layer_head_pairs:
            layer_head_pairs = [(layer, head) for layer in range(att_arr.shape[0]) for head in range(att_arr.shape[1])]

        start_index = 0 if num_last_tokens is None else max(0, len(text) - num_last_tokens)
        html_output = ""

        for layer, head in layer_head_pairs:
            if layer < att_arr.shape[0] and head < att_arr.shape[1]:
                attention = att_arr[layer, head].detach().cpu().numpy()
                normalized_attention = (attention - np.min(attention)) / (np.max(attention) - np.min(attention))

                char_width = 9
                char_height = 18
                replacement_width = char_width * len(next(iter(new_decoded_members.values()))) // 2
                if show_layer_head_info:
                    html_text = f'<div>Layer {layer}, Head {head}</div>'
                else:
                    html_text = ''
                html_text += f'<table cellpadding="0" cellspacing="0" style="border-collapse: collapse; line-height: {char_height}px; margin: 0; padding: 0;">'

                for i in range(start_index, len(text)):
                    html_text += '<tr>'
                    for j in range(len(text)):
                        if text[j] in new_decoded_members:
                            char = new_decoded_members[text[j]]
                            width = replacement_width
                        else:
                            char = text[j]
                            width = char_width
                        color_weight = normalized_attention[i, j]
                        background_color = f'rgba(70, 130, 60, {color_weight})'

                        if j == i:
                            html_text += f'<td style="background-color: {background_color}; color: {text_and_line_color}; width: {width}px; height: {char_height}px; font-size: 12px; font-family: monospace; border: 2px solid {text_and_line_color}; text-align: center; white-space: nowrap; padding: 0; margin: 0;">{char}</td>'
                        elif j < i:
                            html_text += f'<td style="background-color: {background_color}; color: {text_and_line_color}; width: {width}px; height: {char_height}px; font-size: 12px; font-family: monospace; border: 1px solid {text_and_line_color}; text-align: center; white-space: nowrap; padding: 0; margin: 0;">{char}</td>'
                        else:
                            html_text += f'<td style="background-color: rgba(70, 130, 80, 0); color: rgba(70, 130, 80, 0); width: {width}px; height: {char_height}px; font-size: 12px; font-family: monospace; border: 0px solid {text_and_line_color}; text-align: center; white-space: nowrap; padding: 0; margin: 0;"></td>'
                    html_text += '</tr>'
                html_text += '</table><br>'

                html_output += html_text
            else:
                print(f"Invalid layer ({layer}) or head ({head}) index.")

        return html_output

