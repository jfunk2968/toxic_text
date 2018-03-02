
import torch 
import torch.utils.data
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import CountVectorizer
import time
import argparse



def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:02.0f}".format(int(hours),int(minutes),round(seconds))


def get_loader(split):
    idx = [i for i, x in enumerate(splits) if x==0]
    X = train_clean.loc[idx, 'clean_text'].copy()
    y = train_clean.loc[idx, 'toxic'].copy()
    X_bow = csr_matrix(vectorizer.transform(X), dtype=np.float32).todense()
    X_tensor = torch.from_numpy(X_bow)
    y_tensor = torch.LongTensor([int(x) for x in y])
    split_set = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    return(torch.utils.data.DataLoader(split_set, 
    	batch_size=100, 
    	shuffle=False, 
    	num_workers=2))


def get_loader_test(start,end):
    X = test_clean.loc[start:end, 'clean_text'].copy()
    X_bow = csr_matrix(vectorizer.transform(X), dtype=np.float32).todense()
    X_tensor = torch.from_numpy(X_bow)
    sset = torch.utils.data.TensorDataset(X_tensor, torch.zeros(len(X)))
    return(torch.utils.data.DataLoader(sset, 
                                       batch_size=100,
                                       shuffle=False, 
                                       num_workers=1))


def create_submission(netmodel, outfile):

    pred = []
    ids = []

    cuts = np.append(np.arange(0, len(test_clean), 1000), len(test_clean)+1)
    
    for i in range(len(cuts)-1):
    
        loader = get_loader_test(cuts[i], cuts[i+1]-1)

        for data, l in loader:
            inputs = data
            outputs = netmodel(Variable(inputs).cuda())
            pred += list(outputs[:,1].data)
    
    scores = pd.concat([test_clean['id'], pd.Series(pred).apply(lambda x: np.exp(x))], axis=1) 
    scores.columns=['id', 'toxic']
    scores.to_csv(outfile, index=False)

    return None


class BoWClassifier(nn.Module):

    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)


class BoWClassifier2(nn.Module):

    def __init__(self, num_labels, vocab_size):
        super(BoWClassifier2, self).__init__()
        self.linear1 = nn.Linear(vocab_size, 64)
        self.dropout = nn.Dropout()
        self.linear2 = nn.Linear(64, num_labels)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("target")
	parser.add_argument("epochs")
	parser.add_argument("mtype")
	args = parser.parse_args()

	if args.target not in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
		print("Not a valid target variable: ", args.target)
		exit()

	if args.mtype not in ['linear','single']:
		print("Not a valid model type: ", args.mtype)
		exit()

	t_start = time.time()

	train_clean = joblib.load('../../data/train_clean.pckl')
	test_clean = joblib.load('../../data/test_clean.pckl')
	vectorizer = joblib.load('../vectorizer.pckl')
	splits = joblib.load('../splits.pckl')

	early_stopping_epochs = 10

	#for target in ['severe_toxic','obscene','threat','insult','identity_hate']:

	w = train_clean[args.target].value_counts(normalize=True).iloc[::-1]
	C = Variable(torch.FloatTensor([w[1], w[0]])).cuda()


	if args.mtype=='linear':
		model = BoWClassifier(2, 177012)

	elif args.mtype=='single':
		model = BoWClassifier2(2, 177012)

	else:
		exit()

	model.cuda()

	loss_function = nn.NLLLoss(weight=C)
	optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=.001)

	test_accuracy = []

	for epoch in range(int(args.epochs)):
	    
		running_loss = 0.0
		running_obs = 0
	    
		for s in range(10):

			loader = get_loader(s)

			for i, data in enumerate(loader, 0):

				inputs, labels = data
				inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

				optimizer.zero_grad()
				log_probs = model(inputs)

				loss = loss_function(log_probs, labels)
				loss.backward()
				optimizer.step()
	        
				running_loss += loss.data[0]
				running_obs += len(data[1])
	        
			print('[%d, %5d] loss: %.6f' % (epoch, s, running_loss / running_obs))
			running_loss = 0.0
			running_obs = 0
	    
		pred = []
		act = []
	    
		for s in range(10, 15):

			loader = get_loader(s)
	        
			for data in loader:
				inputs, labels = data
				outputs = model(Variable(inputs).cuda())
				pred += list(outputs[:,1].data)
				act += list(labels)

		fpr, tpr, thresholds = roc_curve(act, pred, pos_label=1)
		roc_auc = auc(fpr, tpr)
		roc_auc 

		if epoch% 1 == 0:
			print(epoch, ' - Test AUC: %0.6f  %%' % roc_auc)
		test_accuracy.append(roc_auc)

		try:
			if (len(test_accuracy) - test_accuracy.index(max(test_accuracy))) > early_stopping_epochs:
				print('Early Stopping')
				break
		except:
			pass

	df = pd.DataFrame(data = {'accuracy' :  test_accuracy,
		'epoch':  range(len(test_accuracy))})
	plt.figure()		
	ax = sns.pointplot(x='epoch', y='accuracy', data=df)
	ax.set(xlabel='EPOCH', 
		ylabel='Test Accuracy (AUC)', 
		title=args.target.upper()+' - Linear Fit')
	plt.savefig(args.target+'_'+args.mtype+'_epoch_auc')

	torch.save(model.state_dict(), args.target+'_'+args.mtype+'_model_state')

	create_submission(model, args.target+'_'+args.mtype+'_test_scores.csv')

	print("TOTAL RUN TIME: ", timer(t_start, time.time()))