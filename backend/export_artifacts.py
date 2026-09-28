# Run this at the END of RNN111_.ipynb after training.
import json, torch

with open('word2idx.json','w',encoding='utf-8') as f:
    json.dump(word2idx,f,ensure_ascii=False)

config = {
    'seq_len': seq_len,
    'vocab_size': vocab_size,
    'embedding_dim': 50,
    'hidden_size': 144,
    'classes': encoder.classes_.tolist(),
}
with open('config.json','w',encoding='utf-8') as f:
    json.dump(config,f,indent=2)

torch.save(model.state_dict(),'best_simple_rnn.pt')
print('Exported: best_simple_rnn.pt, word2idx.json, config.json')
