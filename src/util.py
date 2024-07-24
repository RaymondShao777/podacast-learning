import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import corpus
from sentence_transformers import SentenceTransformer, util
import pandas as pd

'''
Generate a numpy array from response embeddings
@param dict1: dictionary containing ID:response pairs
@param model: Model to use to generate the embeddings
@return: 2d numpy array with shape [num_inputs, output_dimension] containing encodings for each response
'''
def encode_to_frame(dict1: dict, model: SentenceTransformer):
    list1 = []
    # generate lists for embeddings
    for key in dict1:
        list1.append(dict1[key])

    # create embeddings for each sentence
    embedding = model.encode(preprocess(list1))
    return pd.DataFrame(embedding, index=[key for key in dict1])

'''
Remove stopwords and punctuation, lemmatize words
@param inputString: String or List to process
@return: Processed string or list of processed strings
'''
def preprocess(inputString: str | list) -> str | list:
    if type(inputString) is str:
        # removes filler words
        inputString = inputString.lower()
        inputString = re.sub(r"\b[ouhm]h?[mh]+\W", "", inputString)
        # removes punctuation and quotes
        inputString = re.sub("[^\s\w',/\[\]]", " ", inputString)
        # removes meaningless phrases
        inputString = inputString.replace("i think", "")
        inputString = inputString.replace("it means", "")

        #tokenizes string
        tokenization = word_tokenize(inputString)
        newTokenization = []
        wnl = WordNetLemmatizer(); # used to lemmatize words
        isStopword = False # bool used to skip 2-segment stop words
        #adds stop words (WE CAN EDIT THIS LIST IF NEEDED)
        stopwords = corpus.stopwords.words('english')
        # stopwords.append("there's") # this is how we would append more!!
        for i in range(len(tokenization)):
            token = tokenization[i]

            # skips 2nd segement of 2 segment stopword (ex. "t" in "isn't")
            if isStopword:
                isStopword = False
                continue

            # combine tokens for 2-token stopwords
            if i != len(tokenization) - 1:
                temp = token + tokenization[i+1]
                if temp in set(stopwords):
                    isStopword = True
                    continue

            # adds token if not stopword/puncuation
            if token not in set(stopwords) and token not in string.punctuation:
                # adds lemmatized word to final string
                newTokenization.append(wnl.lemmatize(token))

        return TreebankWordDetokenizer().detokenize(newTokenization)
    elif type(inputString) is list:
        return [preprocess(sentence) for sentence in inputString]
    else:
        raise ValueError("Input must be either string or list!")

def cos_sim(embeddings1, embeddings2):
    comp = util.cos_sim(embeddings1.to_numpy(), embeddings2.to_numpy())
    return pd.DataFrame(comp.numpy(), columns=embeddings2.index, index=embeddings1.index)
