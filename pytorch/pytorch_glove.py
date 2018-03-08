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
from sklearn.metrics import roc_curve, auc
import time
import argparse



def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:02.0f}".format(int(hours),int(minutes),round(seconds))


def create_submission(netmodel, outfile):

	Xtest = joblib.load('../data/glove_mean_embeddings_test.pckl')
	Xtest_tensor = torch.from_numpy(Xtest)
	outputs = netmodel(Variable(Xtest_tensor).cuda())
	pred = list(outputs[:,1].data)

	test_clean = joblib.load('../data/test_clean.pckl')

	scores = pd.concat([test_clean['id'], pd.Series(pred).apply(lambda x: np.exp(x))], axis=1) 
	scores.columns=['id', args.target]
	scores.to_csv(outfile, index=False)
	return None


class GloveClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, num_labels, dims):
        super(GloveClassifier, self).__init__()
        self.linear1 = nn.Linear(dims, 64)
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
	args = parser.parse_args()

	if args.target not in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:
		print("Not a valid target variable: ", args.target)
		exit()

	t_start = time.time()

	train_clean = joblib.load('../data/train_clean.pckl')

	X = joblib.load('../data/glove_mean_embeddings.pckl')

	splits = joblib.load('splits.pckl')
	train_ids = [ i for i, v in enumerate(splits) if v < 10 ]
	valid_ids = [ i for i, v in enumerate(splits) if v >= 10 ]

	Xtrain = X[train_ids,]
	ytrain = train_clean.loc[train_ids, args.target]

	Xvalid = X[valid_ids,]
	yvalid = train_clean.loc[valid_ids, args.target]

	# create training data loader
	Xtrain_tensor = torch.from_numpy(Xtrain)
	ytrain_tensor = torch.LongTensor([int(x) for x in ytrain])
	train_set = torch.utils.data.TensorDataset(Xtrain_tensor, ytrain_tensor)
	train_loader = torch.utils.data.DataLoader(train_set, 
		batch_size=64, shuffle=True, num_workers=3)

	# create validation data loader
	Xvalid_tensor = torch.from_numpy(Xvalid)
	yvalid_tensor = torch.LongTensor([int(x) for x in yvalid])
	valid_set = torch.utils.data.TensorDataset(Xvalid_tensor, yvalid_tensor)
	valid_loader = torch.utils.data.DataLoader(valid_set, 
		batch_size=64, shuffle=True, num_workers=3)

	early_stopping_epochs = 10

	w = train_clean[args.target].value_counts(normalize=True).iloc[::-1]
	C = Variable(torch.FloatTensor([w[1], w[0]])).cuda()

	model = GloveClassifier(2, 300)
	model.cuda()

	loss_function = nn.NLLLoss(weight=C)
	optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=.001)

	test_accuracy = []
	running_loss = 0

	for epoch in range(int(args.epochs)):

		for i, data in enumerate(train_loader):

			inputs, labels = data
			inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

			optimizer.zero_grad()
			log_probs = model(inputs)

			loss = loss_function(log_probs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.data[0]

			if ((i+1)% 100 == 0):
				print('%d loss: %.6f' % (i+1, running_loss / 100))
				running_loss = 0

		pred = []
		act = []

		for i, data in enumerate(valid_loader):

			inputs, labels = data
			outputs = model(Variable(inputs).cuda())
			pred += list(outputs[:,1].data)
			act += list(labels)

		fpr, tpr, thresholds = roc_curve(act, pred, pos_label=1)
		roc_auc = auc(fpr, tpr)
		roc_auc 

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
		title=args.target.upper())
	plt.savefig('glove_mean/'+args.target+'_epoch_auc')

	torch.save(model.state_dict(), 'glove_mean/'+args.target+'_model_state')

	create_submission(model, 'glove_mean/'+args.target+'_test_scores.csv')

	print("TOTAL RUN TIME: ", timer(t_start, time.time()))