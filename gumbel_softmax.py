import torch
import torch.nn as nn
import torch.nn.functional as F
import json, sys

import numpy as np
from torch.utils.data import Dataset, DataLoader
from CIC_Transformer_torch_model import *
from embeds_in_transformer import Transformer_From_Embedding

torch.set_float32_matmul_precision('highest')

WINDOW_SIZE = 20 # number of messages input as context
NUM_TOK_MSG = 9 # number of tokens per message

'''
    This function creates a batch from the original inputs (ins) and the tokens that we
    want to stitch in (other_tokens)
'''
def get_multitoken_batch(ins, outs, res, other_tokens, model, batch_size):
    _batch = None
    num_other_tokens = 0
    if other_tokens != None:
        num_other_tokens = other_tokens.size(0)
    counter = 1
    for k in range(batch_size):
        orig_embed_before = model.decoder_embedding(ins[k][:(ins.size(1) - (k+1))])[None, :, :]
        if k == 0:
            # the last token is the one we are currently searching
            new_embed = torch.concat([orig_embed_before, res[None, None, :]], dim=1)
            _batch = new_embed
        elif k <= num_other_tokens:
            # the last tokens are ones that have been sampled from the trained distribution
            new_embed = torch.concat([orig_embed_before, res[None, None, :], other_tokens[None, :k, :]], dim=1)
            _batch = torch.concat([_batch, new_embed], dim=0)
        else:
            # the last token(s) are not ones that we sampled, so we need to stitch together:
            # 1. the original input
            # 2. the single token we are searching
            # 3. the tokens from the distribution we have already trained
            # 4. the tokens that would come after all of this
            orig_embed_after = model.decoder_embedding(ins[k][-counter:])[None, :, :]
            if other_tokens != None:
                new_embed = torch.concat([orig_embed_before, res[None, None, :], other_tokens[None, :, :], orig_embed_after], dim=1)
            else:
                new_embed = torch.concat([orig_embed_before, res[None, None, :], orig_embed_after], dim=1)
            _batch = torch.concat([_batch, new_embed], dim=0)
            counter += 1

    return _batch


'''
    Performs the Gumbel-Softmax space search optimization
'''
def multi_token_gsm(params, embeddings, model, model_from_embedding, inputs, outputs, size, num_iters, batch_size, logits, tau=1, verbose=False):
    log_coefficients = logits.detach().clone()
    log_coefficients.requires_grad = True
    learning_rate_init = 0.01
    optimizer = torch.optim.Adam([log_coefficients], lr=learning_rate_init)
    log_coefficients.retain_grad()
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    _inputs = torch.clone(inputs)
    _outputs = torch.clone(outputs)
    
    for i in range(num_iters):
        optimizer.zero_grad()
        
        sampled = F.gumbel_softmax(log_coefficients[0], tau=tau, hard=False)
        rest_res = None
        rest_sampled = None
        if log_coefficients.size(0) > 1:
            rest_sampled = F.gumbel_softmax(log_coefficients[1:], tau=tau, hard=False)
            rest_res = rest_sampled @ embeddings[:size, :]
        res = sampled @ embeddings[:size, :]
        _batch = get_multitoken_batch(_inputs, _outputs, res, rest_res, model, batch_size)
        mod_out = model_from_embedding(_batch, is_casual=True)

        # we are modifying more than one token, so we need to change the tokens in our output to match what we selected
        if rest_sampled != None:
            for i in range(rest_sampled.size(0)):
                _outputs[i] = torch.argmax(rest_sampled, dim=-1)[i]
        
        loss = criterion(mod_out[:, -1, :].view(-1, 384), _outputs[:batch_size])
        loss.backward()
        optimizer.step()
        if verbose and i % 10 == 0:
            print(loss)
    return log_coefficients


'''
    Used to determine the loss of a given batch with substituted tokens (tokens)

    It takes into account the loss of predicting the first token in the subs, as
    well as the loss that would be accrued with these tokens in place
'''
def determine_losses(model, ins, outs, loss_fn, tokens, bsize):
    # first, we will calculate the loss for predicting this token
    ft_loss = None
    with torch.no_grad():
        _out = model(ins[:1])
        ft_loss = loss_fn(_out[:, -1, :], torch.tensor(tokens[0], device=_out.device).unsqueeze(0))
    if tokens[0] == 0:
      ft_loss = torch.tensor(100000) # ignored token
      
    inputs = torch.clone(ins[1:]) # skip forward one so the token at the end of inputs[0] is the token we are checking
    outputs_mod = torch.clone(outs[1:bsize+1])
    for i in range(bsize):
        ctr = 0
        for j in range(min(i + 1, len(tokens))):
            idx = -(i + 1) + j
            if -1 * idx > len(inputs[i]):
                ctr += 1
            else:
                inputs[i][idx] = tokens[ctr]
                ctr += 1
    outputs = None
    with torch.no_grad():
        outputs = model(inputs[:bsize])
    for i in range(len(tokens) - 1):
        outputs_mod[i] = tokens[i+1]
    loss = loss_fn(outputs[:, -1, :].view(bsize, -1), outputs_mod)
    outputs.cpu()
    inputs.cpu()
    del outputs, inputs
    # gc.collect()
    # torch.cuda.empty_cache()
    # return torch.sum(loss) + ft_loss
    return torch.concat([ft_loss, loss], dim=-1)


'''
    Divides num_tasks number of tasks across num_proc number of processes.  It does it in a cyclic pattern,
    so element 0 goes to the first process, element 1 goes to the second one, element 2 to the third one, etc.
    This looks like: load_distributor_round(3, 8) -> [[0,3,6], [1,4,7], [2,5]]
'''
def load_distributor_round(num_proc, num_tasks):
    task_list = [[] for _ in range(num_proc)]
    for i in range(num_tasks):
        task_list[i % num_proc].append(i)
    return task_list

''' 
    Loads the testing data and creates inputs and outputs for the pertubation search.
    This gets everything into just the right format, where we have all of the windows we would ever
    need for inputs and the outputs are the single next token we will need for calcuating a loss value.
    Very convenient, all data-handling should be completed in this function.
'''
def load_dataset_HCRL(params, ids, target_id, num_msgs_set, test_size = 5000, get_original=False):
    OFFSET = 1 # how far into the first batch you have to go to get to the first window that we want; how long we have to wait
               # for the first token we want to slide into the window
    test_data = CANDatasetTxt("data/normal_run_data.txt", 
                            WINDOW_SIZE, 
                            NUM_TOK_MSG, 
                            ids, 
                            params["train_size"] + params["validate_size"], 
                            params["train_size"] + params["validate_size"] + test_size)

    test_dataloader = DataLoader(test_data, batch_size=9, shuffle=False, num_workers=12)
    test_dataloader = DeviceDataLoader(test_dataloader, params["device"])
    target_id = torch.tensor(target_id)
    _data = enumerate(test_dataloader)
    target_msg_sets = []
    for d in _data:
        for ds in target_msg_sets:
            if ds["counter"] > 0:
                ds["msgs"].append(d[1])
                ds["counter"] -= 1
        if d[1][0][1][-1] == target_id:
            target_msg_sets.append(
                {
                    "msgs": [d[1]],
                    "counter": num_msgs_set - 1
                }
            )
    target_msg_sets = [target_msg_sets[i] for i in range(len(target_msg_sets)) if target_msg_sets[i]['counter'] == 0]
    ins_and_outs = []
    for n in range(len(target_msg_sets)):
        _ins = torch.cat(tuple([target_msg_sets[n]['msgs'][i][0] for i in range(num_msgs_set)]), dim=0)[OFFSET:]
        _outs = torch.cat(tuple([target_msg_sets[n]['msgs'][i][1] for i in range(num_msgs_set)]), dim=0)[OFFSET:, -1]
        ins_and_outs.append((_ins, _outs))
    if get_original:
        return ins_and_outs, test_dataloader
    else:
        return ins_and_outs

''' 
    Loads the testing data and creates inputs and outputs for the pertubation search.
    This gets everything into just the right format, where we have all of the windows we would ever
    need for inputs and the outputs are the single next token we will need for calcuating a loss value.
    Very convenient, all data-handling should be completed in this function.
'''
def load_dataset_CIC(params, ids, target_id, num_msgs_set, test_size = 5000):
    OFFSET = 1 # how far into the first batch you have to go to get to the first window that we want; how long we have to wait
               # for the first token we want to slide into the window
    ids = [int(i) for i in ids]
    test_data = CANDatasetCSV("data/decimal_benign.csv", 
                            WINDOW_SIZE, 
                            NUM_TOK_MSG, 
                            ids, 
                            params["train_size"] + params["validate_size"], 
                            params["train_size"] + params["validate_size"] + test_size)

    test_dataloader = DataLoader(test_data, batch_size=9, shuffle=False, num_workers=12)
    test_dataloader = DeviceDataLoader(test_dataloader, params["device"])
    target_id = torch.tensor(target_id)
    _data = enumerate(test_dataloader)
    target_msg_sets = []
    for d in _data:
        for ds in target_msg_sets:
            if ds["counter"] > 0:
                ds["msgs"].append(d[1])
                ds["counter"] -= 1
        if d[1][0][1][-1] == target_id:
            target_msg_sets.append(
                {
                    "msgs": [d[1]],
                    "counter": num_msgs_set - 1
                }
            )
    target_msg_sets = [target_msg_sets[i] for i in range(len(target_msg_sets)) if target_msg_sets[i]['counter'] == 0]
    ins_and_outs = []
    for n in range(len(target_msg_sets)):
        _ins = torch.cat(tuple([target_msg_sets[n]['msgs'][i][0] for i in range(num_msgs_set)]), dim=0)[OFFSET:]
        _outs = torch.cat(tuple([target_msg_sets[n]['msgs'][i][1] for i in range(num_msgs_set)]), dim=0)[OFFSET:, -1]
        ins_and_outs.append((_ins, _outs))
    
    return ins_and_outs


def load_models(path, model_name, ids_path, new_device=None):
    
    # load the parameter dictionary for this model
    params = None
    with open(path + model_name + ".json", "r") as f:
        params = json.load(f)
    with open(ids_path, 'rb') as f:
        ids = np.load(f).tolist()
    if new_device != None:
        params['device'] = new_device # spread the load across the number of available CUDA devices

    # load the original model
    model = Transformer(params["tgt_vocab_size"], params["d_model"], params["num_heads"], params["num_layers"], params["d_ff"], params["max_seq_length"], params["dropout"], params['device'])
    model.load_state_dict(torch.load(path + model_name, map_location=lambda storage, loc: storage, weights_only=True))
    model.eval()
    model.to(params["device"])

    # load the model that is the same model as the one above, but modified to take embeddings instead of tokens
    model_from_embedding = Transformer_From_Embedding(params["tgt_vocab_size"], params["d_model"], params["num_heads"], params["num_layers"], params["d_ff"], params["max_seq_length"], params["dropout"], params['device'])
    model_from_embedding.load_state_dict(torch.load(path + model_name, map_location=lambda storage, loc: storage, weights_only=True), strict=False)
    model_from_embedding.eval()
    model_from_embedding.to(params["device"])

    return model, model_from_embedding, params, ids


'''
    Function to run the Gumbel-Softmax input space search for the selected ID (message_id)
    for the next test_size number of messages.

    It will search all num_tokens number of consecutive combinations that it can for each starting
    location in targ_off.

    For example:
        msg_id = 258
        targ_off = [0,1,2]
        num_tokens = 4
        test_size = 500

    This function will perform the Gumbel-Softmax input space search for instances in the next 500
    messages where the ID == 258.  It will do this for the following token combinations:
        - [0, 1, 2, 3] => targ_off[0]
        - [1, 2, 3, 4] => targ_off[1]
        - [2, 3, 4, 5] => targ_off[2]

    The function is currently set up to then check to see if it can find an adversarial sample from
    the derived distribution.  It will return an array with the number of queries required to find
    and adversarial sample, or -1 if it cannot
'''
def run_gsm_multitoken_HCRL(msg_id, targ_off, num_tokens, test_size):
    
    path = "final_models/"
    model_name = "model_20241028_130645_77" # 32 embedding size
    ids_path = 'ids.npy'
    
    
    torch.manual_seed(6452)
    NUM_ITERATIONS = 125
    TOKEN_DIM = 258
    BATCH_SIZE = 8 # SWITCHED to 8 on 7 Jan 2025 from 32, idea is that we should focus the optimization on a smaller number of sequences that follow since that is what the explored token affects the most
    TARGET_ID = msg_id
    NUM_CUDA_DEVICES = torch.cuda.device_count()
    BASIS = 1 # every process needs to skip forward one to get a payload token at the end of the first window
                # but we still need to have the spot where the first payload token is the last output for determining fitness of our selections
    _outs = None
    fitness_loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduce=False)
    cuda_device = "cuda:" + str(TARGET_ID % NUM_CUDA_DEVICES)
    # call a function to get both the original model and the model that takes embeddings, along with the parameters and the reference list of IDs
    model, model_from_embedding, params, ids = load_models(path, model_name, ids_path, cuda_device)
    # get the data needed for this experiment, which is all input windows that our TARGET_ID will show up in
    data_set = load_dataset_HCRL(params, ids, target_id=TARGET_ID, num_msgs_set=22, test_size = test_size)
    emb_tokens = model.decoder_embedding.weight # grab a reference to the embeddings so we can use this in our matrix multiplication later, too
    accuracies = []
    for d in data_set:
        for targ in targ_off:
            logits_in = None
            log_coe = None
            for i in range(num_tokens).__reversed__():
                _ins = d[0][BASIS + targ + i:]
                _outs = d[1][BASIS + targ + i:]
                tmp_ins = d[0][targ:][i].unsqueeze(0)
                logit_seed = model(tmp_ins)[0][-1][:TOKEN_DIM].unsqueeze(0)
                if logits_in == None:
                    logits_in = logit_seed
                else:
                    logits_in = torch.concat([logit_seed, log_coe])
                log_coe = multi_token_gsm(params, emb_tokens, model, model_from_embedding, _ins, _outs, size=258, num_iters=NUM_ITERATIONS, batch_size=BATCH_SIZE + num_tokens, logits=logits_in)
            
            original_tokens = list(d[0][BASIS + targ + num_tokens-1][-num_tokens:])
            original_loss = torch.sum(determine_losses(model, d[0][targ:], d[1][targ:], fitness_loss_fn, original_tokens, WINDOW_SIZE * NUM_TOK_MSG))
            found_at = 0
            found = False
            while found_at < 100:
                sample = list(torch.argmax(F.gumbel_softmax(log_coe, tau=1, hard=True), dim=-1))
                if sample != original_tokens and torch.sum(determine_losses(model, d[0][targ:], d[1][targ:], fitness_loss_fn, sample, WINDOW_SIZE * NUM_TOK_MSG)) <= original_loss:
                    found = True
                    break
                found_at += 1
            if not found:
                found_at = -1
            accuracies.append(found_at)
    return accuracies

'''
    This function, and the one above it, should have been combined, but alas, it was not.
    See the doc above, as the logic is the same
'''
def run_gsm_multitoken_CIC(msg_id, targ_off, num_tokens, test_size):
    
    path = "final_models/"
    model_name = "model_32_45"
    ids_path = "cic_ids.npy"
    
    
    torch.manual_seed(6452)
    NUM_ITERATIONS = 125
    TOKEN_DIM = 257
    BATCH_SIZE = 8 # SWITCHED to 8 on 7 Jan 2025 from 32, idea is that we should focus the optimization on a smaller number of sequences that follow since that is what the explored token affects the most
    NUM_CUDA_DEVICES = torch.cuda.device_count()
    BASIS = 1 # every process needs to skip forward one to get a payload token at the end of the first window
                # but we still need to have the spot where the first payload token is the last output for determining fitness of our selections
    _outs = None
    fitness_loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduce=False)
    cuda_device = "cuda:" + str(msg_id % NUM_CUDA_DEVICES)
    # call a function to get both the original model and the model that takes embeddings, along with the parameters and the reference list of IDs
    model, model_from_embedding, params, ids = load_models(path, model_name, ids_path, cuda_device)
    # get the data needed for this experiment, which is all input windows that our TARGET_ID will show up in
    data_set = load_dataset_CIC(params, ids, target_id=msg_id, num_msgs_set=22, test_size = test_size)
    emb_tokens = model.decoder_embedding.weight # grab a reference to the embeddings so we can use this in our matrix multiplication later, too
    accuracies = []
    for d in data_set:
        for targ in targ_off:
            logits_in = None
            log_coe = None
            for i in range(num_tokens).__reversed__():
                _ins = d[0][BASIS + targ + i:]
                _outs = d[1][BASIS + targ + i:]
                tmp_ins = d[0][targ:][i].unsqueeze(0)
                logit_seed = model(tmp_ins)[0][-1][:TOKEN_DIM].unsqueeze(0)
                if logits_in == None:
                    logits_in = logit_seed
                else:
                    logits_in = torch.concat([logit_seed, log_coe])
                log_coe = multi_token_gsm(params, emb_tokens, model, model_from_embedding, _ins, _outs, size=TOKEN_DIM, num_iters=NUM_ITERATIONS, batch_size=BATCH_SIZE + num_tokens, logits=logits_in)
            original_tokens = list(d[0][BASIS + targ + num_tokens-1][-num_tokens:])
            original_loss = torch.sum(determine_losses(model, d[0][targ:], d[1][targ:], fitness_loss_fn, original_tokens, WINDOW_SIZE * NUM_TOK_MSG))
            found_at = 0
            found = False
            while found_at < 100:
                sample = list(torch.argmax(F.gumbel_softmax(log_coe, tau=1, hard=True), dim=-1))
                if sample != original_tokens and torch.sum(determine_losses(model, d[0][targ:], d[1][targ:], fitness_loss_fn, sample, WINDOW_SIZE * NUM_TOK_MSG)) <= original_loss:
                    found = True
                    break
                found_at += 1
            if not found:
                found_at = -1
            accuracies.append(found_at)
    return accuracies