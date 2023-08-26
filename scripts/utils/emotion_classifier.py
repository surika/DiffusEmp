import os,sys
sys.path.append('/path-to-repo/diffusemp')
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import \
    BertForSequenceClassification, BertConfig, BertTokenizer
from transformers import AdamW
from construct_dataset.load_dataset import load_dataset
import wandb
from tqdm import tqdm, trange

"""
    Train a BERT intent classifier by Hugging Transformer
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# hyp
EPOCH_NUM = 50
BATCH_SIZE = 32
LR = 1e-5

import torch
from transformers import BertTokenizer

"""
    Preprocess the Empathetic Intents dataset
"""
EMOTIONS = ['afraid', 'angry', 'annoyed', 'anticipating', 'anxious', 'apprehensive', 'ashamed', 'caring', 'confident', 'content', 'devastated', 'disappointed', 'disgusted', 'embarrassed', 'excited', 'faithful', 'furious', 'grateful', 'guilty', 'hopeful', 'impressed', 'jealous', 'joyful', 'lonely', 'nostalgic', 'prepared', 'proud', 'sad', 'sentimental', 'surprised', 'terrified', 'trusting']
E2L = {a: i for (a,i) in zip(EMOTIONS, list(range(32)))}

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_data():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = {}
    for split in ['train', 'dev', 'test']:
        dataset_dict = load_dataset(split)
        contexts, emotions = dataset_dict['context'], dataset_dict['emotion']
        labels = [E2L[e] for e in emotions]   
        encodings = tokenizer(contexts, truncation=True, padding=True)
        dataset[split] = EmotionDataset(encodings, labels)
    return dataset


# data
dataset = get_data()
train_loader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset['dev'], batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, shuffle=False)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')




def train_step(model, optimizer, train_loader, num):

    batch_num = 0
    item_num = 0
    total_loss = 0
    correct_num = 0
    model.train()
    for batch in tqdm(train_loader):

        for key, value in batch.items():
            batch[key] = value.cuda()

        optimizer.zero_grad()
        output = model(**batch)
        loss = output.loss
        total_loss += loss.item()
        loss.backward()

        preds = output.logits.argmax(dim=-1)
        correct_num += (preds == batch['labels']).sum().item()
        optimizer.step()

        batch_num += 1
        item_num += batch['labels'].size(0)

    loss = total_loss / batch_num
    acc = correct_num / item_num
    print('Train %d \t ACC %f \t Loss %f' % (num, acc, loss))


def eval_step(model, eval_loader, valid=True):

    batch_num = 0
    item_num = 0
    total_loss = 0
    correct_num = 0
    vocab_num = {key: 0 for key in range(32)}
    vocab_str = {key: list() for key in range(32)}
    model.eval()
    for batch in eval_loader:

        for key, value in batch.items():
            batch[key] = value.cuda()

        output = model(**batch)
        loss = output.loss
        total_loss += loss.item()

        preds = output.logits.argmax(dim=-1)

        if valid is False:
            for i in range(preds.shape[0]):
                vocab_num[preds[i].item()] += 1
                vocab_str[preds[i].item()].append(tokenizer.decode(batch['input_ids'][i], skip_special_tokens=True))


        correct_num += (preds == batch['labels']).sum().item()
        batch_num += 1
        item_num += batch['labels'].size(0)

    loss = total_loss / batch_num
    acc = correct_num / item_num
    if valid is True:
        print('Valid \t ACC %f \t Loss %f' % (acc, loss))
    else:
        print('Test \t ACC %f \t Loss %f' % (acc, loss))
        print(vocab_num)
        for key, value in vocab_str.items():
            print(key)
            for v in value[: 10]:
                print(v)

    return acc

def main():
    print('Start !')
    wandb.init(
            project=os.getenv("WANDB_PROJECT", "Emotion"),
            name="emotion_origin",
        )

    config = BertConfig.from_pretrained('bert-base-uncased')
    config.num_labels = 32
    model = BertForSequenceClassification(config).cuda()
    optimizer = AdamW(model.parameters(), lr=LR)
    # for param in model.base_model.parameters():
    #     param.requires_grad = False
    best = -1
    for i in trange(EPOCH_NUM):
        train_step(model, optimizer, train_loader, i+1)
        result = eval_step(model, valid_loader, valid=True)
        if result > best:
            best = result
            torch.save(model, './context_emotion_best.pkl')
            print('New Best')

    print('Load best model')
    # model = torch.load('./paras.pkl').cuda()
    model = torch.load('./context_emotion_best.pkl').cuda()
    _ = eval_step(model, test_loader, valid=False)


if __name__ == '__main__':

    main()