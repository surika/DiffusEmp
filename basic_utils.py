import argparse
import torch
import json, os, copy
import time, nltk

from diffusemp import gaussian_diffusion as gd
from diffusemp.gaussian_diffusion import SpacedDiffusion, space_timesteps
from diffusemp.transformer_model import TransformerNetModel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

class myTokenizer():
    """
    Load tokenizer from bert config or defined BPE vocab dict
    """
    ################################################
    ### You can custome your own tokenizer here. ###
    ################################################
    def __init__(self, args):
        if args.vocab == 'bert':
            tokenizer = AutoTokenizer.from_pretrained(args.config_name)
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            # save
            tokenizer.save_pretrained(args.checkpoint_path)
        elif args.vocab == 'vm':
            tokenizer = AutoTokenizer.from_pretrained(args.config_name)
            self.tokenizer = tokenizer
            self.sep_token_id = tokenizer.sep_token_id
            self.pad_token_id = tokenizer.pad_token_id
            self.add_special_tokens_()
            # save
            tokenizer.save_pretrained(args.checkpoint_path)
        else: 
            # load vocab from the path
            print('#'*30, 'load vocab from', args.vocab)
            vocab_dict = {'[START]': 0, '[END]': 1, '[UNK]':2, '[PAD]':3}
            with open(args.vocab, 'r', encoding='utf-8') as f:
                for row in f:
                    vocab_dict[row.strip().split(' ')[0]] = len(vocab_dict)
            self.tokenizer = vocab_dict
            self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
            self.sep_token_id = vocab_dict['[END]']
            self.pad_token_id = vocab_dict['[PAD]']
            # save
            if int(os.environ['LOCAL_RANK']) == 0:
                path_save_vocab = f'{args.checkpoint_path}/vocab.json'
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)
                
        self.vocab_size = len(self.tokenizer)
        args.vocab_size = self.vocab_size # update vocab size in args
    
    def encode_token(self, sentences):
        if isinstance(self.tokenizer, dict):
            input_ids = [[0] + [self.tokenizer.get(x, self.tokenizer['[UNK]']) for x in seq.split()] + [1] for seq in sentences]
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            input_ids = self.tokenizer(sentences, add_special_tokens=True)['input_ids']
        else:
            assert False, "invalid type of vocab_dict"
        return input_ids
        
    def decode_token(self, seq):
        if isinstance(self.tokenizer, dict):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            seq = seq.squeeze(-1).tolist()
            while len(seq)>0 and seq[-1] == self.pad_token_id:
                seq.pop()
            tokens = self.tokenizer.decode(seq)
        else:
            assert False, "invalid type of vocab_dict"
        return tokens
    
    def add_special_tokens_(self, model = None): # from EDGE
        """ Add special tokens to the tokenizer and the model if they have not already been added. """
        # orig_num_tokens = len(tokenizer.encoder)
        # num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
        emotion_word_list = [
            '[surprised]', '[excited]', '[annoyed]', '[proud]', '[angry]', '[sad]', '[grateful]', '[lonely]',
            '[impressed]', '[afraid]', '[disgusted]', '[confident]', '[terrified]', '[hopeful]','[anxious]', '[disappointed]',
            '[joyful]', '[prepared]', '[guilty]', '[furious]', '[nostalgic]', '[jealous]',
            '[anticipating]', '[embarrassed]',
            '[content]', '[devastated]', '[sentimental]', '[caring]', '[trusting]', '[ashamed]',
            '[apprehensive]', '[faithful]']
        other_edge_list = ['[yes]','[no]','[why]','[when]','[who]','[what]','[where]','[which]','[?]','[how]','[pronoun]','[]','[_]']
        intent_word_list = ['[agreeing]','[acknowledging]','[encouraging]','[consoling]','[sympathizing]','[suggesting]','[questioning]','[wishing]','[neutral]']
        cm_word_list = ["[emotional_reaction]", "[exploration]", "[interpretation]"]
        frame_word_list = []
        with open('/path-to-datasets/EmpatheticDialogue/frames_list.txt', encoding='utf-8') as f:
            for sent in f.readlines():
                for word in sent.strip().split(' '):
                    word = "[" + word + "]"
                    frame_word_list.append(word)
        all_special_tokens = emotion_word_list + intent_word_list + cm_word_list + frame_word_list + other_edge_list
        num_added_toks = self.tokenizer.add_tokens(all_special_tokens)
        if model is not None and num_added_toks > 0:
            model.resize_token_embeddings(len(self.tokenizer))

    def encode_token_with_control(self, context, target, emotion, ert, ext, ipt, intent, sf): 
        def compute_mask(context_ids=None, input_id=None, SF_ids=None, IT_ids=None, ER_ids=None, EX_ids=None, IP_ids=None, seg_pos=None):
            # obtain subwords seq
            sp1_ids = [1,1,1]
            sp2_ids = [1]
            input_ids = []
            for id in input_id:
                if len(id) == 1:
                    input_ids.append(id[0])
                elif len(id) > 1:
                    for id_ in id:
                        input_ids.append(id_)
            
            # initial mask vertor for every ids
            mask_ids = [0] * len(context_ids + ER_ids + EX_ids + IP_ids + IT_ids + SF_ids + sp1_ids + input_ids + sp2_ids)
            
            # pos_CM_ids = len(context_ids)
            pos_ER_ids = len(context_ids)
            # pos_IT_ids = pos_ER_ids + len(ER_ids)
            pos_EX_ids = pos_ER_ids + len(ER_ids)
            pos_IP_ids = pos_EX_ids + len(EX_ids)
            pos_IT_ids = pos_IP_ids + len(IP_ids)
            pos_SF_ids = pos_IT_ids + len(IT_ids)
            pos_sp1_ids = pos_SF_ids + len(SF_ids)
            pos_input_ids = pos_sp1_ids + len(sp1_ids)
            pos_sp2_ids = pos_input_ids + len(input_ids)


            # all context_ids and input_ids can see each other, 
            # prepared a temp for the next step
            tmp_context = []
            for i in range(len(context_ids)):
                tmp_context.append(i)
            
            tmp_IT = []
            for i in range(pos_IT_ids, pos_SF_ids):
                tmp_IT.append(i)
            # print(tmp_IT)
            tmp_sf = []
            for i in range(pos_SF_ids, pos_sp1_ids):
                tmp_sf.append(i)
            
            tmp_sp1 = []
            for i in range(pos_sp1_ids, pos_input_ids):
                tmp_sp1.append(i)
            
            tmp_target = []
            for i in range(pos_input_ids, pos_sp2_ids):
                tmp_target.append(i)

            tmp_sp2 = []
            for i in range(pos_sp2_ids, len(mask_ids)):
                tmp_sp2.append(i)



            # memorize the pos of one-hots
            # one_hots_CM = tmp_context + [pos_CM_ids] + tmp_target
            one_hots_ER = tmp_context + [pos_ER_ids] + tmp_sp1 + tmp_target + tmp_sp2
            one_hots_EX = tmp_context + [pos_EX_ids] + tmp_sp1 + tmp_target + tmp_sp2
            one_hots_IP = tmp_context + [pos_IP_ids] + tmp_sp1 + tmp_target + tmp_sp2

            one_hots_IT = []
            IT_TARGET_DICT = {}
            for i in range(len(IT_ids)):
                pos = tmp_context + tmp_IT + tmp_sp1 + tmp_target[seg_pos[i]:seg_pos[i+1]] + tmp_sp2 # + 2
                for target in tmp_target[seg_pos[i]:seg_pos[i+1]]:
                    IT_TARGET_DICT[target] = tmp_IT[i]
                one_hots_IT.append(pos)
            # print('one_hot_it',one_hots_IT)
            # print('IT_TARGET_DICT',IT_TARGET_DICT)

            one_hots_sf = []
            len_sub_word = [0]
            count = 0
            for word in input_id:
                count += len(word)
                len_sub_word.append(count)

            for i in range(len(input_id)):
                one_hots_sf.append(tmp_context + tmp_sf + tmp_sp1 + [a for a in tmp_target[len_sub_word[i]:len_sub_word[i+1]]] + tmp_sp2)

            # print('one_hot_sf',one_hots_sf)

            

            one_hots_target = []
            for i in range(len(input_id)):
                if len(input_id[i]) == 1:
                    rst = tmp_context + [pos_EX_ids-1] + [pos_IP_ids-1] + [pos_IT_ids-1] + [tmp_sf[i]] + tmp_sp1 + tmp_target + tmp_sp2
                    one_hots_target.append(rst)

                elif len(input_id[i]) > 1:
                    for j in range(len(input_id[i])):
                        rst = tmp_context + [pos_EX_ids-1] + [pos_IP_ids-1] + [pos_IT_ids-1] + [tmp_sf[i]] + tmp_sp1 + tmp_target + tmp_sp2
                        one_hots_target.append(rst)

            for i in range(len(one_hots_target)):
                one_hots_target[i] = one_hots_target[i] + [IT_TARGET_DICT[i + pos_input_ids]]

            # print('one_hots_target',one_hots_target)
            attention_mask = []
            for _ in context_ids:
                attention_mask.append([1] * len(context_ids + ER_ids + EX_ids + IP_ids + IT_ids + SF_ids + sp1_ids+input_ids+sp2_ids))
            
            for _ in ER_ids:
                new_mask_ids = copy.deepcopy(mask_ids)
                for pos in one_hots_ER:
                    new_mask_ids[pos] = 1
                attention_mask.append(new_mask_ids)
            for _ in EX_ids:
                new_mask_ids = copy.deepcopy(mask_ids)
                for pos in one_hots_EX:
                    new_mask_ids[pos] = 1
                attention_mask.append(new_mask_ids)
            for _ in IP_ids:
                new_mask_ids = copy.deepcopy(mask_ids)
                for pos in one_hots_IP:
                    new_mask_ids[pos] = 1
                attention_mask.append(new_mask_ids)

            for one_hots in one_hots_IT:
                new_mask_ids1 = copy.deepcopy(mask_ids)
                for pos in one_hots:
                    new_mask_ids1[pos] = 1
                attention_mask.append(new_mask_ids1)

            for one_hots in one_hots_sf:
                new_mask_ids2 = copy.deepcopy(mask_ids)
                for pos in one_hots:
                    new_mask_ids2[pos] = 1
                attention_mask.append(new_mask_ids2)
            
            for _ in range(3): #sp1
                attention_mask.append([1] * len(context_ids + ER_ids + EX_ids + IP_ids + IT_ids + SF_ids + sp1_ids+input_ids+sp2_ids))

            
            for one_hots in one_hots_target:
                new_mask_ids3 = copy.deepcopy(mask_ids)
                for pos in one_hots:
                    new_mask_ids3[pos] = 1
                attention_mask.append(new_mask_ids3)

            for _ in range(1): #sp2
                attention_mask.append([1] * len(context_ids + ER_ids + EX_ids + IP_ids + IT_ids + SF_ids + sp1_ids+input_ids+sp2_ids))

            return attention_mask

        # special token processing
        cls_id = self.tokenizer("[CLS]", add_special_tokens=True)['input_ids'][1:-1]
        sep_id = self.tokenizer("[SEP]", add_special_tokens=True)['input_ids'][1:-1]
        emotion = "[" + emotion + "]"
        context = emotion + " " + context
        ert = "[" + ert + "]"
        ext = "[" + ext + "]"
        ipt = "[" + ipt + "]"
        sf = sf.split()
        sf = ["["+e+"]" for e in sf]
        sf = " ".join(sf)
        
        temp = []
        for it in intent.split():
            temp.append("["+it+"]")
        intents = " ".join(temp)
        intents = intents.split()

        # tokenized ids for context, emp and sf
        # er_ids, ex_ids, ip_ids, it_ids, sf_ids, ids = [], [], [], [], [], [] 
        tokenized_context_id = self.tokenizer(context, add_special_tokens=True)['input_ids'] 
        tokenized_er_id = self.tokenizer(ert, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_ex_id = self.tokenizer(ext, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_ip_id = self.tokenizer(ipt, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_sf_id = self.tokenizer(sf, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_it_id = self.tokenizer(intents, add_special_tokens=True)['input_ids'] 
        tokenized_it_id = [t[1:-1] for t in tokenized_it_id]
        tokenized_input_id = []
        seg_pos = [0]
        # tokenized ids for intent
        for i in range(len(target)):
            temp_target = nltk.tokenize.word_tokenize(target[i].strip())
            input_ids = self.tokenizer(temp_target, add_special_tokens=True)['input_ids']
            input_ids = [t[1:-1] for t in input_ids]
            tokenized_input_id.extend(input_ids)
            temp_ = [z for zz in input_ids for z in zz]
            seg_pos.append(seg_pos[-1]+len(temp_))
            # intent_ids = tokenizer(intents[i], add_special_tokens=True)['input_ids'][1:-1]  if len(intents[i])>0 else [] 
            # for j in range(len(temp_target)):
            #     input_ids = tokenizer(temp_target[j], add_special_tokens=True)['input_ids'][1:-1]
        attention_mask = compute_mask(context_ids=tokenized_context_id, input_id=tokenized_input_id, SF_ids=tokenized_sf_id, IT_ids=tokenized_it_id, ER_ids=tokenized_er_id, EX_ids=tokenized_ex_id, IP_ids=tokenized_ip_id, seg_pos=seg_pos)
        for i in range(len(attention_mask)):
            for j in range(len(attention_mask[i])):
                assert attention_mask[i][j] == attention_mask [j][i]
                    # print('error i',i,'j',j)
        # labels = tokenized_context_id + tokenized_er_id + tokenized_ex_id + tokenized_ip_id + [tt for t in tokenized_it_id for tt in t] + tokenized_sf_id + [102,102,101]+[tt for t in tokenized_input_id for tt in t]+[102]
        # labels2 = tokenizer.convert_ids_to_tokens(labels)
        # labels3 = list(range(len(labels)))
        # labels = [a+str(b) for a,b in zip(labels2,labels3)]
        # fig = px.imshow(attention_mask,x=labels,y=labels, xgap=4,ygap=4)
        # fig.show()
        # for mask in attention_mask:
        #     print(mask)
        input_ids_x = tokenized_context_id + tokenized_er_id + tokenized_ex_id + tokenized_ip_id + [tt for t in tokenized_it_id for tt in t] + tokenized_sf_id + sep_id
        input_ids_y = cls_id + [tt for t in tokenized_input_id for tt in t] + sep_id
            
        return input_ids_x, input_ids_y, attention_mask, len(tokenized_context_id)

    def encode_token_wocm(self, context, target, emotion, intent, sf): 
        def compute_mask(context_ids=None, input_id=None, SF_ids=None, IT_ids=None, seg_pos=None):
            # obtain subwords seq
            sp1_ids = [1,1,1]
            sp2_ids = [1]
            input_ids = []
            for id in input_id:
                if len(id) == 1:
                    input_ids.append(id[0])
                elif len(id) > 1:
                    for id_ in id:
                        input_ids.append(id_)
            
            # initial mask vertor for every ids
            mask_ids = [0] * len(context_ids + IT_ids + SF_ids + sp1_ids + input_ids + sp2_ids)
            
            # pos_ER_ids = len(context_ids)
            # pos_EX_ids = pos_ER_ids + len(ER_ids)
            # pos_IP_ids = pos_EX_ids + len(EX_ids)
            # pos_IT_ids = pos_IP_ids + len(IP_ids)
            pos_IT_ids = len(context_ids)
            pos_SF_ids = pos_IT_ids + len(IT_ids)
            pos_sp1_ids = pos_SF_ids + len(SF_ids)
            pos_input_ids = pos_sp1_ids + len(sp1_ids)
            pos_sp2_ids = pos_input_ids + len(input_ids)


            # all context_ids and input_ids can see each other, 
            # prepared a temp for the next step
            tmp_context = []
            for i in range(len(context_ids)):
                tmp_context.append(i)
            
            tmp_IT = []
            for i in range(pos_IT_ids, pos_SF_ids):
                tmp_IT.append(i)
            print(tmp_IT)
            tmp_sf = []
            for i in range(pos_SF_ids, pos_sp1_ids):
                tmp_sf.append(i)
            
            tmp_sp1 = []
            for i in range(pos_sp1_ids, pos_input_ids):
                tmp_sp1.append(i)
            
            tmp_target = []
            for i in range(pos_input_ids, pos_sp2_ids):
                tmp_target.append(i)

            tmp_sp2 = []
            for i in range(pos_sp2_ids, len(mask_ids)):
                tmp_sp2.append(i)



            # memorize the pos of one-hots
            # one_hots_ER = tmp_context + [pos_ER_ids] + tmp_sp1 + tmp_target + tmp_sp2
            # one_hots_EX = tmp_context + [pos_EX_ids] + tmp_sp1 + tmp_target + tmp_sp2
            # one_hots_IP = tmp_context + [pos_IP_ids] + tmp_sp1 + tmp_target + tmp_sp2

            one_hots_IT = []
            IT_TARGET_DICT = {}
            for i in range(len(IT_ids)):
                pos = tmp_context + tmp_IT + tmp_sp1 + tmp_target[seg_pos[i]:seg_pos[i+1]] + tmp_sp2 # + 2
                for target in tmp_target[seg_pos[i]:seg_pos[i+1]]:
                    IT_TARGET_DICT[target] = tmp_IT[i]
                one_hots_IT.append(pos)

            one_hots_sf = []
            len_sub_word = [0]
            count = 0
            for word in input_id:
                count += len(word)
                len_sub_word.append(count)

            for i in range(len(input_id)):
                one_hots_sf.append(tmp_context + tmp_sf + tmp_sp1 + [a for a in tmp_target[len_sub_word[i]:len_sub_word[i+1]]] + tmp_sp2)        

            one_hots_target = []
            for i in range(len(input_id)):
                if len(input_id[i]) == 1:
                    rst = tmp_context + [tmp_sf[i]] + tmp_sp1 + tmp_target + tmp_sp2
                    one_hots_target.append(rst)

                elif len(input_id[i]) > 1:
                    for j in range(len(input_id[i])):
                        rst = tmp_context + [tmp_sf[i]] + tmp_sp1 + tmp_target + tmp_sp2
                        one_hots_target.append(rst)

            for i in range(len(one_hots_target)):
                one_hots_target[i] = one_hots_target[i] + [IT_TARGET_DICT[i + pos_input_ids]]

            attention_mask = []
            for _ in context_ids:
                attention_mask.append([1] * len(context_ids + IT_ids + SF_ids + sp1_ids+input_ids+sp2_ids))
            
            # for _ in ER_ids:
            #     new_mask_ids = copy.deepcopy(mask_ids)
            #     for pos in one_hots_ER:
            #         new_mask_ids[pos] = 1
            #     attention_mask.append(new_mask_ids)
            # for _ in EX_ids:
            #     new_mask_ids = copy.deepcopy(mask_ids)
            #     for pos in one_hots_EX:
            #         new_mask_ids[pos] = 1
            #     attention_mask.append(new_mask_ids)
            # for _ in IP_ids:
            #     new_mask_ids = copy.deepcopy(mask_ids)
            #     for pos in one_hots_IP:
            #         new_mask_ids[pos] = 1
            #     attention_mask.append(new_mask_ids)

            for one_hots in one_hots_IT:
                new_mask_ids1 = copy.deepcopy(mask_ids)
                for pos in one_hots:
                    new_mask_ids1[pos] = 1
                attention_mask.append(new_mask_ids1)

            for one_hots in one_hots_sf:
                new_mask_ids2 = copy.deepcopy(mask_ids)
                for pos in one_hots:
                    new_mask_ids2[pos] = 1
                attention_mask.append(new_mask_ids2)
            
            for _ in range(3): #sp1
                attention_mask.append([1] * len(context_ids + IT_ids + SF_ids + sp1_ids+input_ids+sp2_ids))

            
            for one_hots in one_hots_target:
                new_mask_ids3 = copy.deepcopy(mask_ids)
                for pos in one_hots:
                    new_mask_ids3[pos] = 1
                attention_mask.append(new_mask_ids3)

            for _ in range(1): #sp2
                attention_mask.append([1] * len(context_ids + IT_ids + SF_ids + sp1_ids+input_ids+sp2_ids))

            return attention_mask

        # special token processing
        cls_id = self.tokenizer("[CLS]", add_special_tokens=True)['input_ids'][1:-1]
        sep_id = self.tokenizer("[SEP]", add_special_tokens=True)['input_ids'][1:-1]
        emotion = "[" + emotion + "]"
        context = emotion + " " + context
        # ert = "[" + ert + "]"
        # ext = "[" + ext + "]"
        # ipt = "[" + ipt + "]"
        sf = sf.split()
        sf = ["["+e+"]" for e in sf]
        sf = " ".join(sf)
        
        temp = []
        for it in intent.split():
            temp.append("["+it+"]")
        intents = " ".join(temp)
        intents = intents.split()

        # tokenized ids for context, emp and sf
        # er_ids, ex_ids, ip_ids, it_ids, sf_ids, ids = [], [], [], [], [], [] 
        tokenized_context_id = self.tokenizer(context, add_special_tokens=True)['input_ids'] 
        # tokenized_er_id = tokenizer(ert, add_special_tokens=True)['input_ids'][1:-1]
        # tokenized_ex_id = tokenizer(ext, add_special_tokens=True)['input_ids'][1:-1]
        # tokenized_ip_id = tokenizer(ipt, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_sf_id = self.tokenizer(sf, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_it_id = self.tokenizer(intents, add_special_tokens=True)['input_ids'] 
        tokenized_it_id = [t[1:-1] for t in tokenized_it_id]
        tokenized_input_id = []
        seg_pos = [0]
        # tokenized ids for intent
        for i in range(len(target)):
            temp_target = nltk.tokenize.word_tokenize(target[i].strip())
            input_ids = self.tokenizer(temp_target, add_special_tokens=True)['input_ids']
            input_ids = [t[1:-1] for t in input_ids]
            tokenized_input_id.extend(input_ids)
            temp_ = [z for zz in input_ids for z in zz]
            seg_pos.append(seg_pos[-1]+len(temp_))
            # intent_ids = tokenizer(intents[i], add_special_tokens=True)['input_ids'][1:-1]  if len(intents[i])>0 else [] 
            # for j in range(len(temp_target)):
            #     input_ids = tokenizer(temp_target[j], add_special_tokens=True)['input_ids'][1:-1]
        attention_mask = compute_mask(context_ids=tokenized_context_id, input_id=tokenized_input_id, SF_ids=tokenized_sf_id, IT_ids=tokenized_it_id, seg_pos=seg_pos)
        for i in range(len(attention_mask)):
            for j in range(len(attention_mask[i])):
                if attention_mask[i][j] != attention_mask [j][i]:
                    print('error i',i,'j',j)
        input_ids_x = tokenized_context_id + [tt for t in tokenized_it_id for tt in t] + tokenized_sf_id + sep_id
        input_ids_y = cls_id + [tt for t in tokenized_input_id for tt in t] + sep_id
            
        return input_ids_x, input_ids_y, attention_mask, len(tokenized_context_id)

    def encode_token_woit(self, context, target, emotion, ert, ext, ipt, sf): 
        def compute_mask(context_ids=None, input_id=None, SF_ids=None, ER_ids=None, EX_ids=None, IP_ids=None):
            # obtain subwords seq
            sp1_ids = [1,1,1]
            sp2_ids = [1]
            input_ids = []
            for id in input_id:
                if len(id) == 1:
                    input_ids.append(id[0])
                elif len(id) > 1:
                    for id_ in id:
                        input_ids.append(id_)
            
            # initial mask vertor for every ids
            mask_ids = [0] * len(context_ids + ER_ids + EX_ids + IP_ids + SF_ids + sp1_ids + input_ids + sp2_ids)
            
            # pos_CM_ids = len(context_ids)
            pos_ER_ids = len(context_ids)
            # pos_IT_ids = pos_ER_ids + len(ER_ids)
            pos_EX_ids = pos_ER_ids + len(ER_ids)
            pos_IP_ids = pos_EX_ids + len(EX_ids)
            pos_SF_ids = pos_IP_ids + len(IP_ids)
            pos_sp1_ids = pos_SF_ids + len(SF_ids)
            pos_input_ids = pos_sp1_ids + len(sp1_ids)
            pos_sp2_ids = pos_input_ids + len(input_ids)


            # all context_ids and input_ids can see each other, 
            # prepared a temp for the next step
            tmp_context = []
            for i in range(len(context_ids)):
                tmp_context.append(i)
            
            # tmp_IT = []
            # for i in range(pos_IT_ids, pos_SF_ids):
            #     tmp_IT.append(i)
            # print(tmp_IT)

            tmp_sf = []
            for i in range(pos_SF_ids, pos_sp1_ids):
                tmp_sf.append(i)
            
            tmp_sp1 = []
            for i in range(pos_sp1_ids, pos_input_ids):
                tmp_sp1.append(i)
            
            tmp_target = []
            for i in range(pos_input_ids, pos_sp2_ids):
                tmp_target.append(i)

            tmp_sp2 = []
            for i in range(pos_sp2_ids, len(mask_ids)):
                tmp_sp2.append(i)



            # memorize the pos of one-hots
            # one_hots_CM = tmp_context + [pos_CM_ids] + tmp_target
            one_hots_ER = tmp_context + [pos_ER_ids] + tmp_sp1 + tmp_target + tmp_sp2
            one_hots_EX = tmp_context + [pos_EX_ids] + tmp_sp1 + tmp_target + tmp_sp2
            one_hots_IP = tmp_context + [pos_IP_ids] + tmp_sp1 + tmp_target + tmp_sp2

            # one_hots_IT = []
            # IT_TARGET_DICT = {}
            # for i in range(len(IT_ids)):
            #     pos = tmp_context + tmp_IT + tmp_sp1 + tmp_target[seg_pos[i]:seg_pos[i+1]] + tmp_sp2 # + 2
            #     for target in tmp_target[seg_pos[i]:seg_pos[i+1]]:
            #         IT_TARGET_DICT[target] = tmp_IT[i]
            #     one_hots_IT.append(pos)
            # print('one_hot_it',one_hots_IT)
            # print('IT_TARGET_DICT',IT_TARGET_DICT)

            one_hots_sf = []
            len_sub_word = [0]
            count = 0
            for word in input_id:
                count += len(word)
                len_sub_word.append(count)

            for i in range(len(input_id)):
                one_hots_sf.append(tmp_context + tmp_sf + tmp_sp1 + [a for a in tmp_target[len_sub_word[i]:len_sub_word[i+1]]] + tmp_sp2)

            # print('one_hot_sf',one_hots_sf)

            

            one_hots_target = []
            for i in range(len(input_id)):
                if len(input_id[i]) == 1:
                    rst = tmp_context + [pos_EX_ids-1] + [pos_IP_ids-1] + [pos_IP_ids] + [tmp_sf[i]] + tmp_sp1 + tmp_target + tmp_sp2
                    one_hots_target.append(rst)

                elif len(input_id[i]) > 1:
                    for j in range(len(input_id[i])):
                        rst = tmp_context + [pos_EX_ids-1] + [pos_IP_ids-1] + [pos_IP_ids] + [tmp_sf[i]] + tmp_sp1 + tmp_target + tmp_sp2
                        one_hots_target.append(rst)

            for i in range(len(one_hots_target)):
                one_hots_target[i] = one_hots_target[i]

            attention_mask = []
            for _ in context_ids:
                attention_mask.append([1] * len(context_ids + ER_ids + EX_ids + IP_ids + SF_ids + sp1_ids+input_ids+sp2_ids))
            
            for _ in ER_ids:
                new_mask_ids = copy.deepcopy(mask_ids)
                for pos in one_hots_ER:
                    new_mask_ids[pos] = 1
                attention_mask.append(new_mask_ids)
            for _ in EX_ids:
                new_mask_ids = copy.deepcopy(mask_ids)
                for pos in one_hots_EX:
                    new_mask_ids[pos] = 1
                attention_mask.append(new_mask_ids)
            for _ in IP_ids:
                new_mask_ids = copy.deepcopy(mask_ids)
                for pos in one_hots_IP:
                    new_mask_ids[pos] = 1
                attention_mask.append(new_mask_ids)

            for one_hots in one_hots_sf:
                new_mask_ids2 = copy.deepcopy(mask_ids)
                for pos in one_hots:
                    new_mask_ids2[pos] = 1
                attention_mask.append(new_mask_ids2)
            
            for _ in range(3): #sp1
                attention_mask.append([1] * len(context_ids + ER_ids + EX_ids + IP_ids + SF_ids + sp1_ids+input_ids+sp2_ids))

            
            for one_hots in one_hots_target:
                new_mask_ids3 = copy.deepcopy(mask_ids)
                for pos in one_hots:
                    new_mask_ids3[pos] = 1
                attention_mask.append(new_mask_ids3)

            for _ in range(1): #sp2
                attention_mask.append([1] * len(context_ids + ER_ids + EX_ids + IP_ids + SF_ids + sp1_ids+input_ids+sp2_ids))

            return attention_mask

        # special token processing
        cls_id = self.tokenizer("[CLS]", add_special_tokens=True)['input_ids'][1:-1]
        sep_id = self.tokenizer("[SEP]", add_special_tokens=True)['input_ids'][1:-1]
        emotion = "[" + emotion + "]"
        context = emotion + " " + context
        ert = "[" + ert + "]"
        ext = "[" + ext + "]"
        ipt = "[" + ipt + "]"
        sf = sf.split()
        sf = ["["+e+"]" for e in sf]
        sf = " ".join(sf)
        
        # temp = []
        # for it in intent.split():
        #     temp.append("["+it+"]")
        # intents = " ".join(temp)
        # intents = intents.split()

        # tokenized ids for context, emp and sf
        # er_ids, ex_ids, ip_ids, it_ids, sf_ids, ids = [], [], [], [], [], [] 
        tokenized_context_id = self.tokenizer(context, add_special_tokens=True)['input_ids'] 
        tokenized_er_id = self.tokenizer(ert, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_ex_id = self.tokenizer(ext, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_ip_id = self.tokenizer(ipt, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_sf_id = self.tokenizer(sf, add_special_tokens=True)['input_ids'][1:-1]
        # tokenized_it_id = tokenizer(intents, add_special_tokens=True)['input_ids']
        # tokenized_it_id = [t[1:-1] for t in tokenized_it_id]
        tokenized_input_id = []
        # seg_pos = [0]
        for i in range(len(target)):
            temp_target = nltk.tokenize.word_tokenize(target[i].strip())
            input_ids = self.tokenizer(temp_target, add_special_tokens=True)['input_ids']
            input_ids = [t[1:-1] for t in input_ids]
            tokenized_input_id.extend(input_ids)
            # temp_ = [z for zz in input_ids for z in zz]
            # seg_pos.append(seg_pos[-1]+len(temp_))
        attention_mask = compute_mask(context_ids=tokenized_context_id, input_id=tokenized_input_id, SF_ids=tokenized_sf_id, ER_ids=tokenized_er_id, EX_ids=tokenized_ex_id, IP_ids=tokenized_ip_id)
        for i in range(len(attention_mask)):
            for j in range(len(attention_mask[i])):
                if attention_mask[i][j] != attention_mask [j][i]:
                    print('error i',i,'j',j)
        # labels = tokenized_context_id + tokenized_er_id + tokenized_ex_id + tokenized_ip_id + tokenized_sf_id + [102,102,101]+[tt for t in tokenized_input_id for tt in t]+[102]
        # labels2 = self.tokenizer.convert_ids_to_tokens(labels)
        # labels3 = list(range(len(labels)))
        # labels = [a+str(b) for a,b in zip(labels2,labels3)]
        # fig = px.imshow(attention_mask,x=labels,y=labels)
        # fig.show()
        # for mask in attention_mask:
        #     print(mask)
        input_ids_x = tokenized_context_id + tokenized_er_id + tokenized_ex_id + tokenized_ip_id + tokenized_sf_id + sep_id
        input_ids_y = cls_id + [tt for t in tokenized_input_id for tt in t] + sep_id
            
        return input_ids_x, input_ids_y, attention_mask, len(tokenized_context_id)

    def encode_token_wosf(self, context, target, emotion, ert, ext, ipt, intent): 
        def compute_mask(context_ids=None, input_id=None, IT_ids=None, ER_ids=None, EX_ids=None, IP_ids=None, seg_pos=None):
            # obtain subwords seq
            sp1_ids = [1,1,1]
            sp2_ids = [1]
            input_ids = []
            for id in input_id:
                if len(id) == 1:
                    input_ids.append(id[0])
                elif len(id) > 1:
                    for id_ in id:
                        input_ids.append(id_)
            
            # initial mask vertor for every ids
            mask_ids = [0] * len(context_ids + ER_ids + EX_ids + IP_ids + IT_ids + sp1_ids + input_ids + sp2_ids)
            
            # pos_CM_ids = len(context_ids)
            pos_ER_ids = len(context_ids)
            # pos_IT_ids = pos_ER_ids + len(ER_ids)
            pos_EX_ids = pos_ER_ids + len(ER_ids)
            pos_IP_ids = pos_EX_ids + len(EX_ids)
            pos_IT_ids = pos_IP_ids + len(IP_ids)
            pos_sp1_ids = pos_IT_ids + len(IT_ids)
            # pos_sp1_ids = pos_SF_ids + len(SF_ids)
            pos_input_ids = pos_sp1_ids + len(sp1_ids)
            pos_sp2_ids = pos_input_ids + len(input_ids)


            # all context_ids and input_ids can see each other, 
            # prepared a temp for the next step
            tmp_context = []
            for i in range(len(context_ids)):
                tmp_context.append(i)
            
            tmp_IT = []
            for i in range(pos_IT_ids, pos_sp1_ids):
                tmp_IT.append(i)
            # print(tmp_IT)
            # tmp_sf = []
            # for i in range(pos_SF_ids, pos_sp1_ids):
            #     tmp_sf.append(i)
            
            tmp_sp1 = []
            for i in range(pos_sp1_ids, pos_input_ids):
                tmp_sp1.append(i)
            
            tmp_target = []
            for i in range(pos_input_ids, pos_sp2_ids):
                tmp_target.append(i)

            tmp_sp2 = []
            for i in range(pos_sp2_ids, len(mask_ids)):
                tmp_sp2.append(i)



            # memorize the pos of one-hots
            # one_hots_CM = tmp_context + [pos_CM_ids] + tmp_target
            one_hots_ER = tmp_context + [pos_ER_ids] + tmp_sp1 + tmp_target + tmp_sp2
            one_hots_EX = tmp_context + [pos_EX_ids] + tmp_sp1 + tmp_target + tmp_sp2
            one_hots_IP = tmp_context + [pos_IP_ids] + tmp_sp1 + tmp_target + tmp_sp2

            one_hots_IT = []
            IT_TARGET_DICT = {}
            for i in range(len(IT_ids)):
                pos = tmp_context + tmp_IT + tmp_sp1 + tmp_target[seg_pos[i]:seg_pos[i+1]] + tmp_sp2 # + 2
                for target in tmp_target[seg_pos[i]:seg_pos[i+1]]:
                    IT_TARGET_DICT[target] = tmp_IT[i]
                one_hots_IT.append(pos)
            # print('one_hot_it',one_hots_IT)
            # print('IT_TARGET_DICT',IT_TARGET_DICT)

            # one_hots_sf = []
            # len_sub_word = [0]
            # count = 0
            # for word in input_id:
            #     count += len(word)
            #     len_sub_word.append(count)

            # for i in range(len(input_id)):
            #     one_hots_sf.append(tmp_context + tmp_sf + tmp_sp1 + [a for a in tmp_target[len_sub_word[i]:len_sub_word[i+1]]] + tmp_sp2)

            # print('one_hot_sf',one_hots_sf)

            

            one_hots_target = []
            for i in range(len(input_id)):
                if len(input_id[i]) == 1:
                    rst = tmp_context + [pos_EX_ids-1] + [pos_IP_ids-1] + [pos_IT_ids-1] + tmp_sp1 + tmp_target + tmp_sp2
                    one_hots_target.append(rst)

                elif len(input_id[i]) > 1:
                    for j in range(len(input_id[i])):
                        rst = tmp_context + [pos_EX_ids-1] + [pos_IP_ids-1] + [pos_IT_ids-1] + tmp_sp1 + tmp_target + tmp_sp2
                        one_hots_target.append(rst)

            for i in range(len(one_hots_target)):
                one_hots_target[i] = one_hots_target[i] + [IT_TARGET_DICT[i + pos_input_ids]]

            # print('one_hots_target',one_hots_target)
            attention_mask = []
            for _ in context_ids:
                attention_mask.append([1] * len(context_ids + ER_ids + EX_ids + IP_ids + IT_ids + sp1_ids+input_ids+sp2_ids))
            
            for _ in ER_ids:
                new_mask_ids = copy.deepcopy(mask_ids)
                for pos in one_hots_ER:
                    new_mask_ids[pos] = 1
                attention_mask.append(new_mask_ids)
            for _ in EX_ids:
                new_mask_ids = copy.deepcopy(mask_ids)
                for pos in one_hots_EX:
                    new_mask_ids[pos] = 1
                attention_mask.append(new_mask_ids)
            for _ in IP_ids:
                new_mask_ids = copy.deepcopy(mask_ids)
                for pos in one_hots_IP:
                    new_mask_ids[pos] = 1
                attention_mask.append(new_mask_ids)

            for one_hots in one_hots_IT:
                new_mask_ids1 = copy.deepcopy(mask_ids)
                for pos in one_hots:
                    new_mask_ids1[pos] = 1
                attention_mask.append(new_mask_ids1)

            # for one_hots in one_hots_sf:
            #     new_mask_ids2 = copy.deepcopy(mask_ids)
            #     for pos in one_hots:
            #         new_mask_ids2[pos] = 1
            #     attention_mask.append(new_mask_ids2)
            
            for _ in range(3): #sp1
                attention_mask.append([1] * len(context_ids + ER_ids + EX_ids + IP_ids + IT_ids + sp1_ids+input_ids+sp2_ids))

            
            for one_hots in one_hots_target:
                new_mask_ids3 = copy.deepcopy(mask_ids)
                for pos in one_hots:
                    new_mask_ids3[pos] = 1
                attention_mask.append(new_mask_ids3)

            for _ in range(1): #sp2
                attention_mask.append([1] * len(context_ids + ER_ids + EX_ids + IP_ids + IT_ids + sp1_ids+input_ids+sp2_ids))

            return attention_mask

        # special token processing
        cls_id = self.tokenizer("[CLS]", add_special_tokens=True)['input_ids'][1:-1]
        sep_id = self.tokenizer("[SEP]", add_special_tokens=True)['input_ids'][1:-1]
        emotion = "[" + emotion + "]"
        context = emotion + " " + context
        ert = "[" + ert + "]"
        ext = "[" + ext + "]"
        ipt = "[" + ipt + "]"
        # sf = sf.split()
        # sf = ["["+e+"]" for e in sf]
        # sf = " ".join(sf)
        
        temp = []
        for it in intent.split():
            temp.append("["+it+"]")
        intents = " ".join(temp)
        intents = intents.split()

        # tokenized ids for context, emp and sf
        # er_ids, ex_ids, ip_ids, it_ids, sf_ids, ids = [], [], [], [], [], [] 
        tokenized_context_id = self.tokenizer(context, add_special_tokens=True)['input_ids'] 
        tokenized_er_id = self.tokenizer(ert, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_ex_id = self.tokenizer(ext, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_ip_id = self.tokenizer(ipt, add_special_tokens=True)['input_ids'][1:-1]
        # tokenized_sf_id = tokenizer(sf, add_special_tokens=True)['input_ids'][1:-1]
        tokenized_it_id = self.tokenizer(intents, add_special_tokens=True)['input_ids'] 
        tokenized_it_id = [t[1:-1] for t in tokenized_it_id]
        tokenized_input_id = []
        seg_pos = [0]
        # tokenized ids for intent
        for i in range(len(target)):
            temp_target = nltk.tokenize.word_tokenize(target[i].strip())
            input_ids = self.tokenizer(temp_target, add_special_tokens=True)['input_ids']
            tokenized_input_id.extend(input_ids)
            temp_ = [z for zz in input_ids for z in zz]
            seg_pos.append(seg_pos[-1]+len(temp_))
            # intent_ids = tokenizer(intents[i], add_special_tokens=True)['input_ids'][1:-1]  if len(intents[i])>0 else [] 
            # for j in range(len(temp_target)):
            #     input_ids = tokenizer(temp_target[j], add_special_tokens=True)['input_ids'][1:-1]
        attention_mask = compute_mask(context_ids=tokenized_context_id, input_id=tokenized_input_id, IT_ids=tokenized_it_id, ER_ids=tokenized_er_id, EX_ids=tokenized_ex_id, IP_ids=tokenized_ip_id, seg_pos=seg_pos)
        for i in range(len(attention_mask)):
            for j in range(len(attention_mask[i])):
                if attention_mask[i][j] != attention_mask [j][i]:
                    print('error i',i,'j',j)
        # labels = tokenized_context_id + tokenized_er_id + tokenized_ex_id + tokenized_ip_id + [tt for t in tokenized_it_id for tt in t] + [102,102,101]+[tt for t in tokenized_input_id for tt in t]+[102]
        # labels2 = tokenizer.convert_ids_to_tokens(labels)
        # labels3 = list(range(len(labels)))
        # labels = [a+str(b) for a,b in zip(labels2,labels3)]
        # fig = px.imshow(attention_mask,x=labels,y=labels)
        # fig.show()
        input_ids_x = tokenized_context_id + tokenized_er_id + tokenized_ex_id + tokenized_ip_id + [tt for t in tokenized_it_id for tt in t] + sep_id
        input_ids_y = cls_id + [tt for t in tokenized_input_id for tt in t] + sep_id
            
        return input_ids_x, input_ids_y, attention_mask, len(tokenized_context_id)

        
def load_model_emb(args, tokenizer):
    ### random emb or pre-defined embedding like glove embedding. You can custome your own init here.
    model = torch.nn.Embedding(tokenizer.vocab_size, args.hidden_dim)
    path_save = '{}/random_emb.torch'.format(args.checkpoint_path)
    path_save_ind = path_save + ".done"
    if int(os.environ['LOCAL_RANK']) == 0:
        if os.path.exists(path_save):
            print('reload the random embeddings', model)
            model.load_state_dict(torch.load(path_save))
        else:
            print('initializing the random embeddings', model)
            torch.nn.init.normal_(model.weight)
            torch.save(model.state_dict(), path_save)
            os.sync()
            with open(path_save_ind, "x") as _:
                pass
    else:
        while not os.path.exists(path_save_ind):
            time.sleep(1)
        print('reload the random embeddings', model)
        model.load_state_dict(torch.load(path_save))

    return model, tokenizer


def load_tokenizer(args):
    tokenizer = myTokenizer(args)

    return tokenizer

def load_defaults_config():
    """
    Load defaults for training args.
    """
    with open('diffusemp/config.json', 'r') as f:
        return json.load(f)


def create_model_and_diffusion(
    hidden_t_dim,
    hidden_dim,
    vocab_size,
    config_name,
    use_plm_init,
    dropout,
    diffusion_steps,
    noise_schedule,
    learn_sigma,
    timestep_respacing,
    predict_xstart,
    rescale_timesteps,
    sigma_small,
    rescale_learned_sigmas,
    use_kl,
    notes,
    **kwargs,
):
    model = TransformerNetModel(
        input_dims=hidden_dim,
        output_dims=(hidden_dim if not learn_sigma else hidden_dim*2),
        hidden_t_dim=hidden_t_dim,
        dropout=dropout,
        config_name=config_name,
        vocab_size=vocab_size,
        init_pretrained=use_plm_init
    )

    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)

    if not timestep_respacing:
        timestep_respacing = [diffusion_steps]

    diffusion = SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        rescale_timesteps=rescale_timesteps,
        predict_xstart=predict_xstart,
        learn_sigmas = learn_sigma,
        sigma_small = sigma_small,
        use_kl = use_kl,
        rescale_learned_sigmas=rescale_learned_sigmas
    )

    return model, diffusion


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")