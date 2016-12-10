from nltk.corpus import stopwords
from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
from nltk.metrics import accuracy
from nltk.metrics import precision
from nltk.metrics import recall
import nltk

f = open('train.tsv', 'r')
fw = open('tmp.tsv', 'w')
tokens = []
scores = []
count = 0
stopwordsSign = ['.', ',']
stopwordsSign += stopwords.words('english')
for line in f:
	cells = line.lower().split('\t')
	sentence = cells[2]
	score = cells[3]
	print count 
	count += 1
	words = []
	for word in sentence.split(' '):
		if word not in stopwordsSign:
			words.append(word)
	fw.write(','.join(words))
	fw.write('\n')
	scores.append(score)
	tokens.append(words)
f.close()
fw.close()

keyset = set()
for token in tokens:
	keyset.update(token)
keys = list(keyset)
print len(keys)
features = []
for token, score in zip(tokens, scores)[:30000]:
	feature = {}
	tokenSet = set(token)
	index = 0
	for key in keys:
		feature[key] = key in tokenSet
		index += 1
	features.append((feature, int(score)))
	print score

trainset, testset = features[:int(len(features)*0.8)], features[int(len(features)*0.8):]
testdata = []
testlabel = []
for testelement in testset:
	testdata.append(testelement[0])
	testlabel.append(testelement[1])
print "start calculate"

classifier = nltk.NaiveBayesClassifier.train(trainset)
testresult = classifier.classify_many(testdata)
reference1 = set([])
test1 = set([])

print "start evaluation"
print("acuracy is " + str(accuracy(testlabel, testresult)))
print "sentiment rate =0:"
for index in range(len(testresult)):
	if testlabel[index] == 0:
		reference1.add(index)
	if testresult[index] == 0:
		test1.add(index)
print("precision is " + str(precision(reference1, test1)))
print("recall is " + str(recall(reference1, test1)))

print "sentiment rate =1:"
for index in range(len(testresult)):
        if testlabel[index] == 1:
                reference1.add(index)
        if testresult[index] == 1:
                test1.add(index)
print("precision is " + str(precision(reference1, test1)))
print("recall is " + str(recall(reference1, test1)))

print "sentiment rate =2:"
for index in range(len(testresult)):
        if testlabel[index] == 2:
                reference1.add(index)
        if testresult[index] == 2:
                test1.add(index)
print("precision is " + str(precision(reference1, test1)))
print("recall is " + str(recall(reference1, test1)))

print "sentiment rate =3:"
for index in range(len(testresult)):
        if testlabel[index] == 3:
                reference1.add(index)
        if testresult[index] == 3:
                test1.add(index)
print("precision is " + str(precision(reference1, test1)))
print("recall is " + str(recall(reference1, test1)))

print "sentiment rate =4:"
for index in range(len(testresult)):
        if testlabel[index] == 4:
                reference1.add(index)
        if testresult[index] == 4:
                test1.add(index)
print("precision is " + str(precision(reference1, test1)))
print("recall is " + str(recall(reference1, test1)))

#print(nltk.classify.accuracy(classifier, testset))

