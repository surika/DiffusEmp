import os, sys, glob, json, math
sys.path.append('..')
import numpy as np
import argparse
import torch

from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2TokenizerFast,  AutoTokenizer, AutoModelForSequenceClassification
from torchmetrics.text.rouge import ROUGEScore
rougeScore = ROUGEScore()
from bert_score import score
from datasets import load_dataset

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from nltk.tokenize import word_tokenize
from epitome.empathy_classifier import EmpathyClassifier

from tqdm import trange, tqdm

def get_cm_score(response_posts, gold_labels=None): #lis
    # gold_er = np.load('/path-to-datasets/moel_empatheticdialogue/sys_er_texts.test.npy', allow_pickle=True)
    # gold_ex = np.load('/path-to-datasets/moel_empatheticdialogue/sys_ex_texts.test.npy', allow_pickle=True)
    # gold_ip = np.load('/path-to-datasets/moel_empatheticdialogue/sys_ip_texts.test.npy', allow_pickle=True)

    ER_model_path = "/path-to-repo/MODELS/ER_with_rationale.pth"
    IP_model_path = "/path-to-repo/MODELS/IP_with_rationale.pth"
    EX_model_path = "/path-to-repo/MODELS/EX_with_rationale.pth"

    seeker_posts = []
    with open("/path-to-repo/DiffusEmp/datasets/EmpatheticDialogue/test.jsonl", 'r') as fs:
        for row in fs:
            source = json.loads(row)['src'].strip()
            source = source.replace(args.eos, '').replace(args.sos, '')
            seeker_posts.append(source)
    seeker_posts = seeker_posts[:len(response_posts)]

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    empathy_classifier = EmpathyClassifier(device,
                            ER_model_path = ER_model_path, 
                            IP_model_path = IP_model_path,
                            EX_model_path = EX_model_path,)

    predictions_ERs, predictions_IPs, predictions_EXs = [], [], []
    # predictions_rationale_ERs, predictions_rationale_IPs, predictions_rationale_EXs = [],[],[]

    for i in trange(len(seeker_posts)):
        (logits_empathy_ER, predictions_ER, \
            logits_empathy_IP, predictions_IP, \
                logits_empathy_EX, predictions_EX, \
                    logits_rationale_ER, predictions_rationale_ER, \
                        logits_rationale_IP, predictions_rationale_IP, \
                            logits_rationale_EX,predictions_rationale_EX) \
                                = empathy_classifier.predict_empathy([seeker_posts[i]], [response_posts[i]])
        predictions_ERs.append(predictions_ER[0])
        predictions_IPs.append(predictions_IP[0])
        predictions_EXs.append(predictions_EX[0])
    return np.mean(predictions_ERs),np.mean(predictions_IPs),np.mean(predictions_EXs)
    # return predictions_ERs, predictions_IPs, predictions_EXs


def get_intent_acc(responses, folder = None):
    gold_labels = []
    if folder is not None:
        temp_path = os.path.split(args.folder)[0].replace('generation_outputs',"diffusion_models")
        config_path = os.path.join(temp_path, "training_args.json")
        with open(config_path, 'rb', ) as f:
            training_args = json.load(f)
        data_dir = os.path.join(training_args["data_dir"],"test_new.jsonl")
        with open(data_dir,'r') as f:
            for row in f:
                s = json.loads(row)['it'].strip()
                gold_labels.append(s)

    intents = [
        'agreeing',
        'acknowledging',
        'encouraging',
        'consoling',
        'sympathizing',
        'suggesting',
        'questioning',
        'wishing',
        'neutral']
    # gold_labels = np.load('/path-to-datasets/moel_empatheticdialogue/sys_intents_texts.test.npy', allow_pickle=True)
    # if gold_labels is None:
    #     gold_labels = []
    #     for s in sources:
    #         a = s.split()
    #         gold_labels.append(a[1])
    #     print(gold_labels)
    recog_bert = torch.load('/path-to-repo/EmpHi-main/intent_prediction/response_intent.pkl').cuda()
    recog_bert.eval()
    intent_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    labels = []
    for text in responses:
        encoded_input = intent_tokenizer(text, return_tensors='pt')
        encoded_input = encoded_input.to('cuda')
        output = recog_bert(**encoded_input)
        preds = output.logits.argmax(dim=-1)
        label = intents[preds]
        labels.append(label)
        # print(label, text)
    results  = []
    for (gl, l) in zip(gold_labels, labels):
        if gl == l: 
            results.append(1)
        else:
            results.append(0)
    return np.sum(results)/len(results)

def get_cola_scores(responses):
    tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
    model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA").to('cuda')
    labels = []
    for text in tqdm(responses):
        encoded_input = tokenizer(text, return_tensors='pt')
        encoded_input = encoded_input.to('cuda')
        output = model(**encoded_input)
        preds = output.logits.argmax(dim=-1)
        labels.append(preds.cpu().numpy())
    # print(labels)
    return np.mean(labels)


def hf_ppl(predictions):
    from evaluate import load
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=predictions, model_id='gpt2-large')
    return results['mean_perplexity']

def calc_ppl(candidates, sources=None):
    device = "cuda"
    model_id = "gpt2-large"
    # model_id = "/path-to-repo/CHECKPOINTS/checkpoints_ft_with_emp/lmft_lm_gpt2_emp_gn1_bp1_gc32_lr2_l200_e10_seed42_cgn1"
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    ppls = []
    for i in trange(len(candidates)):
        input_ids = tokenizer(candidates[i], return_tensors='pt').input_ids
        if sources is not None:
            source_ids = tokenizer(sources[i], return_tensors='pt').input_ids
            whole_text = torch.cat([input_ids, source_ids], 1)
            # whole_text = torch.cat([source_ids, input_ids], 1)
            input_ids = whole_text
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        target_ids = input_ids.clone()
        if sources is not None:
            target_ids[:,:len(source_ids)] = -100
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0]
            ppl = float(torch.exp(log_likelihood).cpu().numpy())
            ppls.append(ppl)
            # print(f"sentence {i}")
            # print(ppl)
    return np.mean(ppls)
        #     res[self.metric_name].append(float(torch.exp(log_likelihood).cpu().numpy()))

        # outputs = model(input_tensor, labels=input_tensor)
        # loss, logits = outputs[0:2]
        # return math.exp(loss.item())

def calc_mmi(candidates, sources):
    # sources = list(np.load('/path-to-datasets/moel_empatheticdialogue/sys_dialog_texts.test.npy', allow_pickle=True))
    # sources = [i for l in sources for i in l ]
    # sources=load_dataset('test')
    from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
    torch.set_grad_enabled(False)
    tokenizer = GPT2Tokenizer('/path-to-repo/MODELS/dialogpt_reverse/vocab.json', '/path-to-repo/MODELS/dialogpt_reverse/merges.txt')
    weights = torch.load('/path-to-repo/MODELS/dialogpt_reverse/dialogpt_small_reverse.pkl')
    # fix misused key value
    weights["lm_head.weight"] = weights["lm_head.decoder.weight"]
    weights.pop("lm_head.decoder.weight", None)
    cfg = GPT2Config.from_json_file('/path-to-repo/MODELS/dialogpt_reverse/config.json')
    model: GPT2LMHeadModel = GPT2LMHeadModel(cfg)
    model.load_state_dict(weights,strict=False)
    device = "cuda"
    model.to(device)
    model.eval()

    mmis = []

    def _score_response(output_token, correct_token): #response, context
        inputs = torch.cat((output_token, correct_token), dim=1)
        mask = torch.full_like(output_token, -100, dtype=torch.long)
        labels = torch.cat((mask, correct_token), dim=1)
        outputs = model(inputs, labels=labels)
        return outputs['loss'].float()
    for i in trange(len(candidates)):
        # result = _get_response(total_input[:, -1:], past)
        input_ids = tokenizer(candidates[i], return_tensors='pt').input_ids.cuda()
        source_ids = tokenizer(sources[i], return_tensors='pt').input_ids.cuda()
        score = _score_response(input_ids, source_ids)
        mmis.append(torch.exp(score).cpu().numpy())
        # mmis.append(mmis + (score,))
    # for i in trange(len(candidates)):
    #     input_ids = tokenizer(candidates[i], return_tensors='pt').input_ids
    #     if sources is not None:
    #         source_ids = tokenizer(sources[i], return_tensors='pt').input_ids
    #         whole_text = torch.cat([input_ids, source_ids], 1)
    #         # whole_text = torch.cat([source_ids, input_ids], 1)
    #         input_ids = whole_text
    #     if torch.cuda.is_available():
    #         input_ids = input_ids.cuda()
    #     target_ids = input_ids.clone()
    #     if sources is not None:
    #         target_ids[:,:len(source_ids)] = -100
    #     if torch.cuda.is_available():
    #         input_ids = input_ids.cuda()
    #     with torch.no_grad():
    #         outputs = model(input_ids, labels=target_ids)
    #         log_likelihood = outputs[0]
    #         mmi = float(torch.exp(log_likelihood).cpu().numpy())
    #         mmis.append(mmi)
    return np.mean(mmis)

def calc_distinct_n(n, candidates, print_score: bool = True): #from CEM
    dict = {}
    total = 0
    candidates = [word_tokenize(candidate) for candidate in candidates]
    for sentence in candidates:
        for i in range(len(sentence) - n + 1):
            ney = tuple(sentence[i : i + n])
            dict[ney] = 1
            total += 1
    score = len(dict) / (total + 1e-16)

    if print_score:
        print(f"***** Distinct-{n}: {score*100} *****")

    return score


def calc_distinct(candidates, print_score: bool = True):#from CEM
    scores = []
    for i in range(4):
        score = calc_distinct_n(i + 1, candidates, print_score)
        scores.append(score)
    return scores

def get_bleu(recover, reference):
    return sentence_bleu([recover.split()], reference.split(), smoothing_function=SmoothingFunction().method4,)

def selectBest(sentences):
    selfBleu = [[] for i in range(len(sentences))]
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            score = get_bleu(s1, s2)
            selfBleu[i].append(score)
    for i, s1 in enumerate(sentences):
        selfBleu[i][i] = 0
    idx = np.argmax(np.sum(selfBleu, -1))
    return sentences[idx]

def get_bleu(recover, reference):
    return sentence_bleu([recover.split()], reference.split(), smoothing_function=SmoothingFunction().method4,)

def diversityOfSet(sentences):
    selfBleu = []
    # print(sentences)
    for i, sentence in enumerate(sentences):
        for j in range(i+1, len(sentences)):
            # print(sentence, sentences[j])
            score = get_bleu(sentence, sentences[j])
            selfBleu.append(score)
    if len(selfBleu)==0:
        selfBleu.append(0)
    div4 = distinct_n_gram_inter_sent(sentences, 4)
    return np.mean(selfBleu), div4


def distinct_n_gram(hypn,n):
    dist_list = []
    for hyp in hypn:
        hyp_ngrams = []
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
        total_ngrams = len(hyp_ngrams)
        unique_ngrams = len(list(set(hyp_ngrams)))
        if total_ngrams == 0:
            return 0
        dist_list.append(unique_ngrams/total_ngrams)
    return  np.mean(dist_list)


def distinct_n_gram_inter_sent(hypn, n):
    hyp_ngrams = []
    for hyp in hypn:
        hyp_ngrams += nltk.ngrams(hyp.split(), n)
    total_ngrams = len(hyp_ngrams)
    unique_ngrams = len(list(set(hyp_ngrams)))
    if total_ngrams == 0:
        return 0
    dist_n = unique_ngrams/total_ngrams
    return  dist_n

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='decoding args.')
    parser.add_argument('--folder', type=str, default='', help='path to the folder of decoded texts')
    parser.add_argument('--mbr', action='store_true', help='mbr decoding or not')
    parser.add_argument('--sos', type=str, default='[CLS]', help='start token of the sentence')
    parser.add_argument('--eos', type=str, default='[SEP]', help='end token of the sentence')
    parser.add_argument('--sep', type=str, default='[SEP]', help='sep token of the sentence')
    parser.add_argument('--pad', type=str, default='[PAD]', help='pad token of the sentence')

    args = parser.parse_args()

    files = sorted(glob.glob(f"{args.folder}/*json"))
    # files = sorted(glob.glob(f"{args.folder}*json"))
    sample_num = 0
    with open(files[0], 'r') as f:
        for row in f:
            sample_num += 1

    sentenceDict = {}
    referenceDict = {}
    sourceDict = {}
    for i in range(sample_num):
        sentenceDict[i] = []
        referenceDict[i] = []
        sourceDict[i] = []

    div4 = []
    selfBleu = []
    
    #source
    sources, references=[], []
    with open('/path-to-datasets/BASELINE_RESULTS/cem.json','r') as f:
        for row in f:
            s = json.loads(row)['source'].strip()
            r = json.loads(row)['reference'].strip()
            s.replace(args.eos, '').replace(args.sos, '')
            r.replace(args.eos, '').replace(args.sos, '')
            sources.append(s)
            references.append(r)

    for path in files:
        print(path)
        # sources = []
        # references = []
        recovers = []
        bleu = []
        rougel = []
        avg_len = []
        dist1, dist2, dist3, dist4 = [],[],[],[]
        gold_er, gold_ex, gold_ip, gold_it = [], [], [], []

        with open(path, 'r') as f:
            cnt = 0
            for row in f:
                row_dict = json.loads(row)

                # source = json.loads(row)['source'].strip()
                # reference = json.loads(row)['reference'].strip()
                recover = json.loads(row)['recover'].strip()
                # source = source.replace(args.eos, '').replace(args.sos, '')
                # reference = reference.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '')
                recover = recover.replace(args.eos, '').replace(args.sos, '').replace(args.sep, '').replace(args.pad, '')

                # sources.append(source)
                # references.append(reference)
                recovers.append(recover)

                if "er" in row_dict.keys():
                    gold_er.append(row_dict["er"])
                    gold_ex.append(row_dict["ex"])
                    gold_ip.append(row_dict["ip"])
                    gold_it.append(row_dict["it"])
            
                avg_len.append(len(recover.split(' ')))
                bleu.append(get_bleu(recover, references[cnt]))
                rougel.append(rougeScore(recover, references[cnt])['rougeL_fmeasure'].tolist())
                # dist1.append(distinct_n_gram([recover], 1))
                # dist2.append(distinct_n_gram([recover], 2))
                # dist3.append(distinct_n_gram([recover], 3))
                # dist4.append(distinct_n_gram([recover], 4))

                sentenceDict[cnt].append(recover)
                referenceDict[cnt].append(references[cnt])
                sourceDict[cnt].append(sources[cnt])
                cnt += 1
        
        references = references[:len(recovers)]
        sources = sources[:len(recovers)]
        #
        # P, R, F1 = score(recovers, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
        # ers, ips, exs = get_cm_score(recovers)
        # calc_distinct(recovers)
        acc = get_intent_acc(recovers, args.folder)
        # ppl = calc_ppl(recovers,sources)
        # # ppl = calc_ppl(sources)
        # cola = get_cola_scores(recovers)
        # # ppl = hf_ppl(recovers)
        # mmi = calc_mmi(recovers, sources)

        print('*'*30)
        print('avg intent acc', acc)
        # print('avg ppl', ppl)
        # print('avg mmi', mmi)
        # print('cola', cola)
        # print('avg BLEU score', np.mean(bleu))
        # print('avg berscore', torch.mean(F1))
        # print('avg dist1-4 score', np.mean(dist1),np.mean(dist2),np.mean(dist3),np.mean(dist4))
        # print('avg er/ex/ip', np.mean(ers), np.mean(exs), np.mean(ips))
        # print('avg ROUGE-L score', np.mean(rougel))
        # print('avg len', np.mean(avg_len))

    if len(files)>1:
        if not args.mbr:
            print('*'*30)
            print('Compute diversity...')
            print('*'*30)
            for k, v in sentenceDict.items():
                if len(v) == 0:
                    continue
                sb, d4 = diversityOfSet(v)
                selfBleu.append(sb)
                div4.append(d4)

            print('avg selfBleu score', np.mean(selfBleu))
            print('avg div4 score', np.mean(div4))
        
        else:
            print('*'*30)
            print('MBR...')
            print('*'*30)
            bleu = []
            rougel = []
            avg_len = []
            dist1 = []
            recovers = []
            references = []
            sources = []


            for k, v in sentenceDict.items():
                if len(v) == 0 or len(referenceDict[k]) == 0:
                    continue

                recovers.append(selectBest(v))
                references.append(referenceDict[k][0])
                sources.append(sourceDict[k][0])

            for (source, reference, recover) in zip(sources, references, recovers):
                bleu.append(get_bleu(recover, reference))
                rougel.append(rougeScore(recover, reference)['rougeL_fmeasure'].tolist())
                avg_len.append(len(recover.split(' ')))
                dist1.append(distinct_n_gram([recover], 1))

            # print(len(recovers), len(references), len(recovers))
            
            P, R, F1 = score(recovers, references, model_type='microsoft/deberta-xlarge-mnli', lang='en', verbose=True)
            ers, ips, exs = get_cm_score(references, recovers)
            print('*'*30)
            print('avg BLEU score', np.mean(bleu))
            print('avg ROUGE-l score', np.mean(rougel))
            print('avg berscore', torch.mean(F1))
            print('avg dist1 score', np.mean(dist1))