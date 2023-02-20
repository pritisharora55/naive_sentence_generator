# imports go here
import sys
import numpy as np
from collections import defaultdict
import random
from math import exp, log

"""
Name: Pritish Arora
NUID: 002781425
"""


# Feel free to implement helper functions

class LanguageModel:
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing):
        """Initializes an untrained LanguageModel
        Parameters:
          n_gram (int): the n-gram order of the language model to create
          is_laplace_smoothing (bool): whether or not to use Laplace smoothing
        """
        # instantiating n_grams and laplace smoothing flag
        self.n_gram = n_gram
        self.is_laplace = is_laplace_smoothing

    def make_ngrams(self, tokens: list, n: int) -> list:
        """Creates n-grams for the given token sequence.
        Args:
        tokens (list): a list of tokens as strings
        n (int): the length of n-grams to create

        Returns:
        list: list of tuples of strings, each tuple being one of the individual n-grams
        n = 2
        [(token1, token2), (token2, token3), ....]
        """
        grams = list()
        for i in range(0, len(tokens) - n + 1):
            tup = tuple(tokens[i:n + i])
            grams.append(tup)
            # print(grams)
        return grams

    def count_freq(self, col: list) -> dict:
        """Creates a dictionary with frequency as values and strings as keys.
            Args:
            col (list): a list of columns

            Returns:
            dict: dictionary which can be used to look up frequency of a token or an n_gram
        """
        directory = dict()
        for element in col:
            if element in directory:
                directory[element] += 1
            else:
                directory[element] = 1
        return directory

    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Parameters:
          training_file_path (str): the location of the training data to read

        Returns:
        None
        """
        # read from file
        file = open(training_file_path, "r").read()
        # sentence segmentation
        segmented = file.split('\n')
        # tokenize the data
        tokenized = file.split()
        tokenize_segmented = [i.split() for i in segmented]
        # create a vocabulary
        vocab = list()
        # count frequencies of tokens in vocabulary
        vocab_with_freq = self.count_freq(tokenized)
        # add <UNK> to vocab
        # vocab_with_freq['<UNK>'] = 0
        # Removing less frequent words from vocabulary
        for element in vocab_with_freq:
            if vocab_with_freq[element] == 1:
                vocab.append('<UNK>')
                continue
            else:
                vocab.append(element)
        # replacing the tokens that are not in vocabulary by <UNK>
        for sentence in tokenize_segmented:
            for word in range(len(sentence)):
                if sentence[word] not in vocab:
                    sentence[word] = '<UNK>'

        self.vocab = vocab
        self.vocab_with_freq =vocab_with_freq

        ngramlist = list()
        priorgramlist = list()

        #calculating n_grams and n-1_grams used for calculating probability look up dictionary

        for sentence in tokenize_segmented:
            list1 = self.make_ngrams(sentence, self.n_gram)
            ngramlist.extend(list1)
            list2 = self.make_ngrams(sentence, self.n_gram - 1)
            priorgramlist.extend(list2)

        priorgram = dict()
        # creating frequency dictionaries of n_grams
        ngram = self.count_freq(ngramlist)
        if self.n_gram != 1:
            priorgram = self.count_freq(priorgramlist)
        else:
            priorgram[tuple()] = len(tokenized)

        # calulating probabilities for n_grams and storing in a dictionary
        probgram = dict()
        self.priorgram = priorgram

        for i in ngram:
            j = list(i)
            j = tuple(j[0:self.n_gram - 1])
            if self.is_laplace is True:
                if self.n_gram == 1:
                    prob = (ngram[i] + 1) / (len(set(vocab)) + sum(vocab_with_freq.values()))
                else:
                    prob = (ngram[i] + 1) / (priorgram[j] + len(vocab))
            else:
                prob = ngram[i] / priorgram[j]
            probgram[i] = prob
        self.probgram = probgram

        pass

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of

        Returns:
          float: the probability value of the given string for this model
        """
        # replacing unknown tokens with <UNK>
        tokens = sentence.split()
        for w in range(len(tokens)):
            if tokens[w] not in self.vocab:
                tokens[w] = self.UNK

        # calculating n_grams of the input sentence
        ngrams = self.make_ngrams(tokens, self.n_gram)
        self.N_size_grams = len(ngrams)

        # calculating conditional probabilities as products of probabilities of n_grams
        prod = 1.0
        for n in ngrams:
            if n in self.probgram:
                prod = prod * self.probgram[n]
            else:
                if self.is_laplace is True:
                    n_1 = list(n)
                    n_1 = tuple(n_1[0:self.n_gram - 1])
                    if n_1 in self.priorgram:
                        prod = prod * (1 / (self.priorgram[n_1] + len(self.vocab)))
                    else:
                        prod = prod * (1/ len(self.vocab))
                else:
                    prod = 0.0
        return prod

    def generate_grams(self, seed):

        """ Performs random sampling based on a seed value for n_grams (Shannon Method)
            Parameters:
            seed (tuple): a part of n_gram which need to be sampled
            Returns:
            Randomly sampled n_gram for a given seed value    """

        genlist = list()
        genlist2 = list()
        for i in self.probgram.items():
            if self.n_gram != 1:
                # print(i[0][:(self.n_gram-1)])
                if i[0][:(self.n_gram - 1)] == seed:
                    genlist.append(i[0])
                    genlist2.append(i[1])
            else:
                if i[0][0] != self.SENT_BEGIN:
                    genlist.append(i[0])
                    genlist2.append(i[1])
        r = random.choices(genlist, weights=genlist2)
        return r

    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          str: the generated sentence
        """
        sentence = list()

        if self.n_gram != 1:
            mfactor = (self.n_gram - 1)
        else:
            mfactor = 1

        prep_seed = '<s> ' * (mfactor)
        seed = tuple(prep_seed.split())

        prep_terminate = '</s> ' * (mfactor)
        terminate = tuple(prep_terminate.split())

        # terminate = ('</s>','</s>')

        while seed != terminate:
            words = self.generate_grams(seed)
            sentence.extend(words)
            seed = words[0][-(self.n_gram - 1):]  ##checkkk
        # print(sentence)

        if self.n_gram != 1:
            sent = str()
            for w in sentence:
                sent = sent + ' ' + w[0]
            sent = sent + (' </s>' * (self.n_gram - 1))
        else:
            sent = '<s>'
            for w in sentence:
                sent = sent + ' ' + w[0]
        return sent

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
        Parameters:
          n (int): the number of sentences to generate

        Returns:
          list: a list containing strings, one per generated sentence
        """
        sentences = list()
        for i in range(n):
            sentences.append(self.generate_sentence())
        return sentences

    def perplexity(self, test_sequence):
        """
            Measures the perplexity for the given test sequence with this trained model.
            As described in the text, you may assume that this sequence may consist of many sentences "glued together".

        Parameters:
          test_sequence (string): a sequence of space-separated tokens to measure the perplexity of
        Returns:
          float: the perplexity of the given sequence
        """
        prob = self.score(test_sequence)
        perplexity = (1/prob)**(1/self.N_size_grams)
        return perplexity


def main():
    # TODO: implement
    training_path = sys.argv[1]
    testing_path1 = sys.argv[2]
    testing_path2 = sys.argv[3]

    model1 = LanguageModel(1, True)
    model2 = LanguageModel(2, True)

    test = "<s> <s> <s> oh i increase the walking distance i can go fifteen minutes from icsi </s> </s> </s> \n <s> <s> <s> oh i increase the walking distance </s> </s> </s>"
    # print(test)
    model1.train(training_path)
    model2.train(training_path)

    file1 = open(testing_path1, "r")
    file2 = open(testing_path2, "r")

    list_of_probs_uni1 = list()
    list_of_probs_uni2 = list()
    list_of_probs_bi1 = list()
    list_of_probs_bi2 = list()

    print('Model: unigram, laplace smoothed')
    print('Sentences:')

    uni_sents = model1.generate(10)
    for i in uni_sents:
        print(i)

    c = 0
    for f in file1:
        c = c + 1
        #print(f)
        list_of_probs_uni1.append(model1.score(f))
    print("test corpus: '{0}'".format(testing_path1))
    print('Num of test sentences: ', c)
    print('Average probability: ', np.mean(list_of_probs_uni1))
    print('Standard deviation: ', np.std(list_of_probs_uni1))

    d = 0
    for f in file2:
        d = d + 1
        list_of_probs_uni2.append(model1.score(f))
    print("test corpus: '{0}'".format(testing_path2))
    print('Num of test sentences: ', d)
    print('Average probability: ', np.mean(list_of_probs_uni2))
    print('Standard deviation: ', np.std(list_of_probs_uni2))

    print('****************************')

    file1 = open(testing_path1, "r")
    file2 = open(testing_path2, "r")

    print('Model: bigram, laplace smoothed')
    print('Sentences:')

    bi_sents = model2.generate(10)
    for j in bi_sents:
        print(j)

    c = 0
    for f in file1:
        c = c + 1
        list_of_probs_bi1.append(model2.score(f))
    print("test corpus: '{0}'".format(testing_path1))
    print('Num of test sentences: ', c)
    print('Average probability: ', np.mean(list_of_probs_bi1))
    print('Standard deviation: ', np.std(list_of_probs_bi1))

    d = 0
    for f in file2:
        d = d + 1
        #print(f)
        list_of_probs_bi2.append(model2.score(f))
    print("test corpus: '{0}'".format(testing_path2))
    print('Num of test sentences: ', d)
    print('Average probability: ', np.mean(list_of_probs_bi2))
    print('Standard deviation: ', np.std(list_of_probs_bi2))


    file1 = open(testing_path1, "r")
    file2 = open(testing_path2, "r")

    test_seq_file1 = str()
    test_seq_file2 = str()
    for i in range(10):
        test_seq_file1 += file1.readline()
        test_seq_file2 += file2.readline()



    print("# Perplexity for 1-grams:")

    print('{0}:{1}'.format(testing_path1, model1.perplexity(test_seq_file1)))
    print('{0}:{1}'.format(testing_path2, model1.perplexity(test_seq_file2)))

    print("# Perplexity for 2-grams:")

    print('{0}:{1}'.format(testing_path1, model2.perplexity(test_seq_file1)))
    print('{0}:{1}'.format(testing_path2, model2.perplexity(test_seq_file2)))


if __name__ == '__main__':

    main()

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) != 4:
        print("Usage:", "python lm.py berp-training-four.txt testingfile1.txt testingfile2.txt")
        sys.exit(1)
