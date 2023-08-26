import csv
import pandas as pd
import argparse
import codecs

import torch
from empathy_classifier import EmpathyClassifier

'''
Example:
'''

parser = argparse.ArgumentParser("Test")
parser.add_argument("--input_path", type=str, default="/path-to-datasets/moel_empatheticdialogue/id_seeker_response.tsv", help="path to input test data")
parser.add_argument("--output_path", type=str, default="/path-to-repo/Empathy-Mental-Health/dataset/sample_test_output.csv", help="output file path")

parser.add_argument("--ER_model_path", type=str, default="/path-to-repo/MODELS/ER_with_rationale.pth", help="path to ER model")
parser.add_argument("--IP_model_path", type=str, default="/path-to-repo/MODELS/EX_with_rationale.pth", help="path to IP model")
parser.add_argument("--EX_model_path", type=str, default="/path-to-repo/MODELS/IP_with_rationale.pth", help="path to EX model")

args = parser.parse_args()

if torch.cuda.is_available():
	device = torch.device("cuda")
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")


input_df = pd.read_csv(args.input_path,delimiter='\t',header=0)

ids = input_df.id.astype(str).tolist()
seeker_posts = input_df.seeker_post.astype(str).tolist()
response_posts = input_df.response_post.astype(str).tolist()

empathy_classifier = EmpathyClassifier(device,
						ER_model_path = args.ER_model_path, 
						IP_model_path = args.IP_model_path,
						EX_model_path = args.EX_model_path,)


output_file = codecs.open(args.output_path, 'w', 'utf-8')
csv_writer = csv.writer(output_file, delimiter=',', quotechar='"')

csv_writer.writerow(['id','seeker_post','response_post','ER_label','IP_label','EX_label'])

for i in range(len(seeker_posts)):
	(logits_empathy_ER, predictions_ER, logits_empathy_IP, predictions_IP, logits_empathy_EX, predictions_EX) = empathy_classifier.predict_empathy([seeker_posts[i]], [response_posts[i]])

	csv_writer.writerow([ids[i], seeker_posts[i], response_posts[i], predictions_ER[0], predictions_IP[0], predictions_EX[0]])

output_file.close()
# #Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style) with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`.
# /path-to-repo/miniconda3/envs/intent/lib/python3.8/site-packages/transformers/tokenization_utils_base.py:2149: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
#   warnings.warn(