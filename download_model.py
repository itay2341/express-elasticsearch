import torch
from transformers import AutoTokenizer, AutoModel

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

dummy_input = tokenizer("This is a sample input", return_tensors="pt")
torch.onnx.export(model, 
                  (dummy_input['input_ids'], dummy_input['attention_mask']), 
                  "all-MiniLM-L6-v2.onnx", 
                  input_names=['input_ids', 'attention_mask'], 
                  output_names=['last_hidden_state'], 
                  dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                                'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                                'last_hidden_state': {0: 'batch_size', 1: 'sequence'}})