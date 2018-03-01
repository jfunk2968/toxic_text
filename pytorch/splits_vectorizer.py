
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from random import randint
import joblib



if __name__ == "__main__":

	train_clean = joblib.load('../data/train_clean.pckl')
	test_clean = joblib.load('../data/test_clean.pckl')

	vectorizer = CountVectorizer(ngram_range=(1,1),
	                             tokenizer=None, 
	                             preprocessor=None, 
	                             lowercase=False)

	vectorizer.fit(train_clean['clean_text'])  

	splits = [randint(0, 15) for x in range(len(train_clean))]

	joblib.dump(vectorizer, 'vectorizer.pckl')
	joblib.dump(splits, 'splits.pckl')
