# this is helpful. thank you-> this, is, help, -ful, . , thank ,you
# by response-level
import copy
# context_ids = [34, 46, 42, 78]  # dialogue context + emotion

# CM_ids = [203]
# IP_ids = []
# EX_ids = []

# IT_ids = [201, 202]
# sf_ids = [[101], [102], [103], [105], [106], [108]]

# input_id = [[1],[2],[3,4],[5],[6],[7]]
# encode_token_with_control(target=input_id, ert = CM_ids, ext = EX_ids, ipt = IP_ids, it=IT_ids, sf=sf_ids)

def encode_token_with_control(target, ert, ext, ipt, it, sf): #
    for i in range(len(target)):
        # input_er_id = self.tokenizer(ert[i], add_special_tokens=False)['input_ids']
        # input_ex_id = self.tokenizer(ext[i], add_special_tokens=False)['input_ids']
        # input_ip_id = self.tokenizer(ipt[i], add_special_tokens=False)['input_ids']
        tokenized_er_id = ert[i]
        tokenized_ex_id = ext[i] 
        tokenized_ip_id = ipt[i] 
        er_ids, ex_ids, ip_ids, it_ids, sf_ids, ids = [], [], [], [], [], [] #id
        for j in range(len(target[i])):
            # input_it_ids = self.tokenizer(it[i][j], add_special_tokens=False)['input_ids'] # 9
            # input_sf_ids = self.tokenizer(sf[i][j].strip(), add_special_tokens=False)['input_ids'] list [[6],[7],[8]]
            # input_ids = self.tokenizer(allsubsentences[i][j].strip(), add_special_tokens=False)['input_ids'] #listlist[[1],[2],[3,5]]
            input_it_ids = it[i][j] #
            input_sf_ids = sf[i][j] #list
            input_ids = target[i][j] #list
            subsentence_input_ids, subsentence_sf_ids = [], []
            # for i in input_ids, input_sf_ids):
            for k in input_ids:
                # [3,5] [8]
                # len_ = len(i)
                sf_ids = [input_sf_ids][0] * len(input)
                subsentence_input_ids.extend(input_ids)
                subsentence_sf_ids.extend(sf_ids)
            subsentence_it_ids = [input_it_ids] * len(subsentence_input_ids)

            it_ids.append(subsentence_it_ids)
            sf_ids.append(subsentence_sf_ids)
            ids.append(subsentence_input_ids)
        # er_ids.append([tokenized_er_id] * len(ids))
        # ex_ids.append([tokenized_ex_id] * len(ids))
        # ip_ids.append([tokenized_ip_id] * len(ids))
        edges = {}
        id2position = {}
        ### Target
        for j in ids:
            temp = tokenized_er_id + tokenized_ex_id + tokenized_ip_id + it_ids[j] + sf_ids[j]
            edges[j] = ids + temp
            for k in temp:
                if k in edges.keys():
                    edges[k].append(j)
                else:
                    edges[k] = [j]
        print(edges)

        ### CM
        # for j in er_ids:
        #     edges[j] = ex_ids+ip_ids+ids
        # for j in ex_ids:
        #     edges[j] = er_ids+ip_ids+ids
        # for j in ip_ids:
        #     edges[j] = er_ids+ex_ids+ids
        # ### IT
        # for j in it_ids:
        #     edges[j] = it

def compute_mask(context_ids, input_id, sf_ids, IT_ids, CM_ids, seg_pos):
    
    # obtain subwords seq
    input_ids = []
    for id in input_id:
        if len(id) == 1:
            input_ids.append(id[0])
        elif len(id) > 1:
            for id_ in id:
                input_ids.append(id_)
    
    # initial mask vertor for every ids
    mask_ids = [0] * len(context_ids + CM_ids + IT_ids + sf_ids + input_ids)
    
    pos_CM_ids = len(context_ids)
    pos_IT_ids = pos_CM_ids + len(CM_ids)
    pos_sf_ids = pos_IT_ids + len(IT_ids)
    pos_input_ids = pos_sf_ids + len(sf_ids)


    # all context_ids and input_ids can see each other, 
    # prepared a temp for the next step
    tmp_context = []
    for i in range(len(context_ids)):
        tmp_context.append(i)
    
    tmp_IT = []
    for i in range(pos_IT_ids+1, pos_sf_ids):
        tmp_IT.append(i)
    
    tmp_sf = []
    for i in range(pos_sf_ids+1, pos_input_ids):
        tmp_sf.append(i)
    
    tmp_target = []
    for i in range(pos_input_ids+1, len(mask_ids)):
        tmp_target.append(i)
    
    one_hots_CM = tmp_context + [pos_CM_ids] + tmp_target

    one_hots_IT = []
    for i in range(len(IT_ids)):
        pos = tmp_context + tmp_IT + tmp_target[seg_pos[i]:seg_pos[i+1]]
        one_hots_IT.append(pos)

    # memorize the pos of one-hots
    one_hots_target = []
    one_hots_sf = []
    for i in range(len(input_id)):
        if len(input_id[i]) == 1:
            rst = tmp_context + [pos_CM_ids] + tmp_IT + tmp_sf + tmp_target
            one_hots_target.append(rst)
            one_hots_sf.append(tmp_context + tmp_sf + [tmp_target[i]])

        elif len(input_id[i]) > 1:
            for j in range(len(input_id[i])):
                rst = tmp_context + [pos_CM_ids] + tmp_IT + tmp_sf + tmp_target
                one_hots_target.append(rst)
                one_hots_sf.append(tmp_context + tmp_sf + [tmp_target[i]])

    attention_mask = []
    for _ in context_ids:
        attention_mask.append([1] * len(context_ids + CM_ids + IT_ids + sf_ids + input_ids))
    
    for _ in CM_ids:
        new_mask_ids = copy.deepcopy(mask_ids)
        for pos in one_hots_CM:
            new_mask_ids[pos] = 1
        attention_mask.append(new_mask_ids)

    for _ in IT_ids:
        new_mask_ids1 = copy.deepcopy(mask_ids)
        for poss in one_hots_IT:
            for pos in poss:
                new_mask_ids1[pos] = 1
        attention_mask.append(new_mask_ids1)

    for _ in sf_ids:
        new_mask_ids2 = copy.deepcopy(mask_ids)
        for poss in one_hots_sf:
            for pos in poss:
                new_mask_ids2[pos] = 1
        attention_mask.append(new_mask_ids2)
    
    for _ in input_ids:
        new_mask_ids3 = copy.deepcopy(mask_ids)
        for poss in one_hots_target:
            for pos in poss:
                new_mask_ids3[pos] = 1
        attention_mask.append(new_mask_ids3)
    
    return attention_mask

context_ids = [34, 46, 42, 78]  # dialogue context + emotion

CM_ids = [203]
IP_ids = []
EX_ids = []

IT_ids = [201, 202]
sf_ids = [[101], [102], [103], [105], [106], [108]]
input_id = [[3,4],[1],[2],[5],[6],[7]]
encode_token_with_control(target=[input_id], ert = [CM_ids], ext = [EX_ids], ipt = [IP_ids], it=[IT_ids], sf=[sf_ids])

# seg_pos = [0,4,6]
# attention_mask = compute_mask(context_ids, input_id, sf_ids, IT_ids, CM_ids, seg_pos)
# for mask in attention_mask:
#     print(mask)