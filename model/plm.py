import torch
from transformers import AutoModel, AutoTokenizer


plm_models = {
    'mbert':  'bert-base-multilingual-cased',
    'enbert': 'bert-base-cased',
    'nbbert': 'NbAiLab/nb-bert-base',
    'ptbert': 'neuralmind/bert-base-portuguese-cased',
    'xlm':    'microsoft/infoxlm-base',
    'gpt2':   'gpt2'
}


def load_plm_model(plm_model_name, device):
    lib_plm_model_name = plm_models[plm_model_name]
    print('Loading PLM: ', lib_plm_model_name)
    plm_model = AutoModel.from_pretrained(lib_plm_model_name).to(device)
    plm_tokenizer = AutoTokenizer.from_pretrained(lib_plm_model_name)
    return plm_model, plm_tokenizer


def get_plm_embeddings(plm_model, plm_tokenizer, df_items, title_naming, device, max_encoding_length=50, batch_size=1024):
    print('Getting embeddings for article titles using PLM')
    print('df_items shape:', df_items.shape)

    encoded_inputs = plm_tokenizer(list(df_items[title_naming]), max_length=max_encoding_length, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_inputs['input_ids'].to(device)
    print('input_ids shape:', input_ids.shape)

    feat_vectors = []
    for i in range(0, len(input_ids), batch_size):
        print(f'Progress step: {i+1} / {len(input_ids)//batch_size + 1}')
        encoded_inputs_batch = input_ids[i:, :] if i+batch_size >= len(input_ids) else input_ids[i:i+batch_size, :]
        with torch.no_grad():
            plm_output = plm_model(encoded_inputs_batch)[1]
        feat_vectors.append(plm_output)
    
    feat_tensor = torch.cat(feat_vectors, 0)
    return feat_tensor


def get_items_features(plm_model_name, df_items, title_naming, device, feat_tensor_file=None, max_encoding_length=50, batch_size=1024):
    plm_model, plm_tokenizer = load_plm_model(plm_model_name, device)
    feat_tensor = get_plm_embeddings(plm_model, plm_tokenizer, df_items, title_naming, device, max_encoding_length, batch_size)
    print('Output feat_tensor shape:', feat_tensor.shape)
    if feat_tensor_file is not None: torch.save(feat_tensor, feat_tensor_file)
    return feat_tensor

