# this is helpful. thank you-> this, is, help, -ful, . , thank ,you
# by response-level
import copy, itertools
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
import nltk
# context_ids = [34, 46, 42, 78]  # dialogue context + emotion
data = {"src": "My friend's boyfriend recently made a pass at me. I'm married, and I also am really good friend's with his girlfriend so I felt really bad. I made sure he knew I was loyal and honest to both!Thats horrible. Did you tell on him?I struggled with it, but my friend was in a really bad spot, with no income and raising her grandchild, he was the financial supporter. So, I didn't tell her, as I knew it would really hurt her in several ways, plus she really loves him!", "trg": ["If your best friend is truly a best friend of yours, they won't leave you over this.","It isn't right to tell his girlfriend in my opinion, but it is good if you speak to him about the situation."], "er": "emotional reaction", "ex": "", "ip": "", "sf": "Required_event Personal_relationship Required_event Personal_relationship pronoun _ pronoun _ Departing pronoun _ _ Correctness Telling _ pronoun Opinion pronoun Desirability Statement", "it": ["neutral","acknowledging"]}
# context_ids = data["src"]
# target = data["trg"]
# # alltarget = data['alltrg']
# ert = "emotional_reaction"
# ext = data["ex"]
# ipt = data["ip"]
# sf = "_ _ Required_event Personal_relationship _ _ _ Required_event Personal_relationship _ _ _ _ _ _ Departing _ Proportional_quantity _ _ _ _ _ Correctness _ Telling _ _ _ _ Opinion _ _ _ _ Desirability _ _ Statement _ _ _ _ _ _"
# it = data["it"]

target = ['helpfulness', 'thank']
sf = 'Required_event _'
ert = "emotional_reaction"
ext = ""
ipt = ""
it = data["it"]
# CM_ids = [203]
# IP_ids = []
# EX_ids = []

# IT_ids = [201, 202]
# sf_ids = [[101], [102], [103], [105], [106], [108]]

# input_id = [[1],[2],[3,4],[5],[6],[7]]
# encode_token_with_control(target=input_id, ert = CM_ids, ext = EX_ids, ipt = IP_ids, it=IT_ids, sf=sf_ids)
# target = ["", ""]
def add_special_tokens_(tokenizer, model = None): # EDGE
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    # orig_num_tokens = len(tokenizer.encoder)
    # num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
    emotion_word_list = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 'sad': 5, 'grateful': 6, 'lonely': 7,
            'impressed': 8, 'afraid': 9, 'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15,
            'joyful': 16, 'prepared': 17, 'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
            'anticipating': 22, 'embarrassed': 23,
            'content': 24, 'devastated': 25, 'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31}.keys()
    emotion_word_list = list(emotion_word_list)
    # other_edge_list = ['Yes','No','Why','When','Who','What','Where','Which','?','How','Pronoun']
    other_edge_list = ['pronoun']
    intent_word_list = ['agreeing','acknowledging','encouraging','consoling','sympathizing','suggesting','questioning','wishing','neutral']
    cm_word_list = ["emotional_reaction", "exploration", "interpretation"]
    frame_word_list = []
    with open('/path-to-repo/DiffusEmp/datasets/EmpatheticDialogue/frames_list.txt', encoding='utf-8') as f:
        for sent in f.readlines():
            for word in sent.strip().split(' '):
                frame_word_list.append(word)
    all_special_tokens = emotion_word_list+intent_word_list+cm_word_list+frame_word_list+other_edge_list
    num_added_toks = tokenizer.add_tokens(all_special_tokens)
    if model is not None and num_added_toks > 0:
        model.resize_token_embeddings(len(tokenizer))

def encode_token_with_control(context, target, ert, ext, ipt, it, sf): #
    er_ids, ex_ids, ip_ids, it_ids, sf_ids, ids = [], [], [], [], [], [] #id
    tokenized_context_id = tokenizer(context, add_special_tokens=True)['input_ids'] 
    tokenized_er_id = tokenizer(ert, add_special_tokens=False)['input_ids'] 
    tokenized_ex_id = tokenizer(ext, add_special_tokens=False)['input_ids']
    tokenized_ip_id = tokenizer(ipt, add_special_tokens=False)['input_ids']
    concat_input_ids, split_input_ids, concat_it_ids, union_it_ids = [], [], [], []
    for i in range(len(target)):
        temp_target = nltk.tokenize.word_tokenize(target[i].strip())
        input_it_ids = tokenizer(it[i], add_special_tokens=False)['input_ids']  if len(it[i])>0 else [] # 9
        subsentence_input_ids = []
        for j in range(len(temp_target)):
            input_ids = tokenizer(temp_target[j], add_special_tokens=False)['input_ids']#listlist[[1],[2],[3,5]]
            subsentence_input_ids.append(input_ids)
        temp = [t for l in subsentence_input_ids for t in l]
        subsentence_it_ids = input_it_ids * len(temp)

        #iinten 
        union_it_ids.append((i, len(subsentence_it_ids)))
        
        concat_it_ids.extend(subsentence_it_ids)
        concat_input_ids.extend(temp)
        split_input_ids.extend(subsentence_input_ids)

    if len(sf) == 0:
        sf = ['_']*len(concat_input_ids) 
    else:
        sf = sf.split()
    
    concat_sf_ids, union_sf_ids = [], []
    input_sf_ids = tokenizer(sf, add_special_tokens=False)['input_ids']list [[6],[7],[8]]
    for i in range(len(input_sf_ids)):
        concat_sf_ids.extend(input_sf_ids[i] * len(split_input_ids[i]))
        union_sf_ids.append((i, len(split_input_ids[i])))

    allid = tokenized_er_id + tokenized_ex_id + tokenized_ip_id + concat_it_ids + concat_sf_ids + concat_input_ids

    position2id = {i:a for a, i in zip(allid, range(len(allid)))}
    same_sf = max([len(list(v)) for k,v in itertools.groupby(concat_sf_ids)])
    ### Target
    vertice = [(i,a) for i, a in zip(range(len(allid)), allid)]
    vertice_index = 0
    er_vertice = [(i,a) for i, a in zip(range(vertice_index, vertice_index+len(tokenized_er_id)), tokenized_er_id)]
    vertice_index += len(er_vertice)
    ex_vertice = [(i,a) for i, a in zip(range(vertice_index, vertice_index+len(tokenized_ex_id)), tokenized_ex_id)]
    vertice_index += len(ex_vertice)
    ip_vertice = [(i,a) for i, a in zip(range(vertice_index, vertice_index+len(tokenized_ip_id)), tokenized_ip_id)]
    vertice_index += len(ip_vertice)
    it_vertice_index = vertice_index
    it_vertice = [(i,a) for i, a in zip(range(vertice_index, vertice_index+len(concat_it_ids)), concat_it_ids)]
    vertice_index += len(it_vertice)

    sf_vertice_index = vertice_index
    sf_vertice = [(i,a) for i, a in zip(range(vertice_index, vertice_index+len(concat_sf_ids)), concat_sf_ids)]
    vertice_index += len(sf_vertice)
    target_vertice = [(i,a) for i, a in zip(range(vertice_index, vertice_index+len(concat_input_ids)), concat_input_ids)]
    control_vertice = er_vertice + ex_vertice + ip_vertice + it_vertice + sf_vertice

    union_it = []
    it_front_len = it_vertice_index
    for (i, l) in union_it_ids:
        union_it.append(list(range(it_front_len, it_front_len+l)))
        it_front_len += l
    
    union_sf = []
    sf_front_len = sf_vertice_index
    for (i, l) in union_sf_ids:
        union_sf.append(list(range(sf_front_len, sf_front_len+l)))
        sf_front_len += l
    
    union = union_it + union_sf
    

    target_edges = {}
    control_edges = {}
    for j in range(len(target_vertice)):
        control_vertice_j = tokenized_er_id + tokenized_ex_id + tokenized_ip_id + [concat_it_ids[j]] + [concat_sf_ids[j]]
        target_edges[target_vertice[j]] = target_vertice + er_vertice + ex_vertice + ip_vertice + [it_vertice[j]] + [sf_vertice[j]]
    for k, v in target_edges.items():
        for vi in v:
            if vi not in control_edges.keys():
                control_edges[vi] = [k]
            else:
                control_edges[vi].extend([k])
    for i in range(len(union)):
        if len(union[i]) == 1: continue
        else:
            temp = []
            for ui in union[i]:
                temp.append((ui, position2id[ui]))
            union[i] = temp
    union_reverse_dict = {}
    for i in range(len(union)):
        if len(union[i]) == 1:continue
        else:
            a = union[i][0]
            for bi in range(1, len(union[i])):
                b = union[i][bi]
                union_reverse_dict[b] = a
    for k,v in target_edges.items():
        for vi in range(len(v)):
            if v[vi] in union_reverse_dict.keys():
                v[vi] = union_reverse_dict[v[vi]]
            target_edges[k] = v
    for k,v in control_edges.items():
        if k in er_vertice: control_edges[k] += er_vertice
        if k in ex_vertice: control_edges[k] += ex_vertice
        if k in ip_vertice: control_edges[k] += ip_vertice
        if k in it_vertice: control_edges[k] += it_vertice
        if k in sf_vertice: control_edges[k] += sf_vertice
        if k in target_edges.keys(): control_edges[k] = target_edges[k]
    for k,v in control_edges.items(): #
        for vi in range(len(v)):
            if v[vi] in union_reverse_dict.keys():
                v[vi] = union_reverse_dict[v[vi]]
            control_edges[k] = v
        if k in union_reverse_dict.keys():
            control_edges[union_reverse_dict[k]]+=control_edges[k]
            control_edges[k] = []
    for k,v in control_edges.items():
        control_edges[k] = list(set(control_edges[k]))
    new_control_edges = {}
    for k,v in control_edges.items():
        if len(v)>0: new_control_edges[k] = v
    old_vertices = sorted(new_control_edges.keys())
    new_vertices = []
    for i in range(len(old_vertices)):
        old = old_vertices[i]
        new_vertices.append((i, old[1]))
    old2new = {k:v for k,v in zip(old_vertices, new_vertices)}

    edge = {} # control+target
    for k,v in control_edges.items(): #
        if k in old2new.keys():
            ki = old2new[k]
            edge[ki] = []
            for vi in range(len(v)):
                edge[ki].append(old2new[v[vi]])
    input_ids_ctrl_target = [v[1] for v in sorted(edge.keys())]
    sep_between_control_target = -len(concat_input_ids)
    input_ids_ctrl = input_ids_ctrl_target[:-len(concat_input_ids)]
    input_ids_x = tokenized_context_id + input_ids_ctrl + [tokenizer.sep_token_id]
    # input_ids_ctrl_sep_target = input_ids_ctrl_target.insert(sep_between_control_target, tokenizer.sep_token_id)
    # input_ids_c =[tokenizer.cls_token_id] + [v[1] for v in sorted(edge.keys())] + [tokenizer.sep_token_id] 
    input_ids_y = [tokenizer.cls_token_id] + concat_input_ids + [tokenizer.sep_token_id]
    edge_index = {}
    context_len = len(tokenized_context_id)
    for k,v in edge.items():
        nk = k[0] + context_len
        nv = [vi[0] + context_len for vi in v] + list(range(context_len))
        edge_index[nk] = sorted(nv)
    control_target_len = len(edge_index.keys())
    for i in range(context_len):
        # temp = context_len + control_target_len
        edge_index[i] = list(range(context_len + control_target_len))
    # context_len = len(tokenized_context_id)
    total_len = len(edge_index)
    mask = []
    sep_between_control_target = -len(concat_input_ids)
    sep_between_x_y = sep_between_control_target + 1
    cls_target = sep_between_control_target + 2
    sep_target = -1
    for i in range(total_len):
        temp = [0] * total_len
        for j in edge_index[i]:
            temp[j] = 1
        temp.insert(sep_between_control_target, 1)
        temp.insert(sep_between_control_target, 1)
        temp.insert(sep_between_control_target, 1)
        temp.append(1)
        mask.append(temp)
    mask.insert(sep_between_control_target, [1]*(total_len+4))
    mask.insert(sep_between_control_target, [1]*(total_len+4))
    mask.insert(sep_between_control_target, [1]*(total_len+4))
    mask.append([1]*(total_len+4))
    input_ids_c = edge_index
    # sep_between_control_target = -len(concat_input_ids)
    
    return input_ids_x, input_ids_y, mask, len(tokenized_context_id)


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
add_special_tokens_(tokenizer=tokenizer)
encode_token_with_control("test new", target, ert, ext, ipt, it, sf)

# seg_pos = [0,4,6]
# attention_mask = compute_mask(context_ids, input_id, sf_ids, IT_ids, CM_ids, seg_pos)
# for mask in attention_mask:
#     print(mask)