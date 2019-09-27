# @author: Adel del Valle - adel.delvalle@upr.edu
# Mutacion en Sede: Código sociolinguístico el cual mira la variación y distribución
# del homónimo introducido en las zonas seleccionadas

import tweepy
import os, os.path
import numpy as np
from matplotlib import pyplot as plt
from tkinter import *
import tensorflow as tf
import keras
import nltk
from nltk import word_tokenize
from gensim.models import Word2Vec
import gensim
from sklearn.manifold import TSNE
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.corpus import cess_esp as cess
from nltk import UnigramTagger as ut
from nltk import BigramTagger as bt
import nltk.data
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import TaggedCorpusReader
from nltk.stem import PorterStemmer
from itertools import chain

#Descargas requeridas por la libreria NLTK para ejecutar ciertos métodos
nltk.download('stopwords')
nltk.download('omw')
nltk.download('averaged_perceptron_tagger')
nltk.download('cess_esp')

# Getters del input del usuario.
def getE1():
    return E1.get()


def getE2():
    return E2.get()


def getE3():
    return E3.get()


def getE4():
    return E4.get()


# Funcion que importa la data por zona geografica y la escribe en un archivo
def getData():
    getE3()
    paisUno = getE3()
    getE4()
    paisDos = getE4()
    getE1()
    keyword = getE1()

    getE2()
    numberOfTweets = getE2()
    numberOfTweets = int(numberOfTweets)

    if str(paisUno) == "Puerto Rico":
        with open("corpus.txt", "w+", encoding="utf-8") as corpus:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="18.2208328,-66.5901489,500km").items(
                    numberOfTweets):
                corpus.write(tweet.text + '\n')
            corpus.close()
    elif str(paisUno) == "Mexico":
        with open("corpus.txt", "w+", encoding="utf-8") as corpus:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="23.6345005,-102.5527878,150km").items(
                    numberOfTweets):
                corpus.write(tweet.text)
            corpus.close()
    elif str(paisUno) == "Venezuela":
        with open("corpus.txt", "w+", encoding="utf-8") as corpus:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="6.4237499,-66.5897293,150km").items(
                    numberOfTweets):
                corpus.write(tweet.text)
            corpus.close()
    elif str(paisUno) == "Colombia":
        with open("corpus.txt", "w+", encoding="utf-8") as corpus:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="4.570868,-74.2973328,150km").items(
                    numberOfTweets):
                corpus.write(tweet.text)
            corpus.close()
    elif str(paisUno) == "Argentina":
        with open("corpus.txt", "w+", encoding="utf-8") as corpus:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="-38.4160957,-63.6166725,150km").items(
                    numberOfTweets):
                corpus.write(tweet.text)
            corpus.close()
    elif str(paisUno) == "Costa Rica":
        with open("corpus.txt", "w+", encoding="utf-8") as corpus:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="9.7489166,-83.7534256,150km").items(
                    numberOfTweets):
                corpus.write(tweet.text)
            corpus.close()
    elif str(paisUno) == "Guatemala":
        with open("corpus.txt", "w+", encoding="utf-8") as corpus:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="6.4237499,-66.5897293,150km").items(
                    numberOfTweets):
                corpus.write(tweet.text)
            corpus.close()

    if str(paisDos) == "Puerto Rico":
        with open("corpus2.txt", "w+", encoding="utf-8") as corpus2:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="18.2208328,-66.5901489,150km").items(
                    numberOfTweets):
                corpus2.write(tweet.text)
            corpus2.close()
    elif str(paisDos) == "Mexico":
        with open("corpus2.txt", "w+", encoding="utf-8") as corpus2:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="18.2208328,-66.5901489,500km").items(
                    numberOfTweets):
                corpus2.write(tweet.text)
            corpus2.close()
    elif str(paisDos) == "Venezuela":
        with open("corpus2.txt", "w+", encoding="utf-8") as corpus2:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="6.4237499,-66.5897293,150km").items(
                    numberOfTweets):
                corpus2.write(tweet.text)
            corpus2.close()
    elif str(paisDos) == "Colombia":
        with open("corpus2.txt", "w+", encoding="utf-8") as corpus2:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="4.570868,-74.2973328,150km").items(
                    numberOfTweets):
                corpus2.write(tweet.text)
            corpus2.close()
    elif str(paisDos) == "Argentina":
        with open("corpus2.txt", "w+", encoding="utf-8") as corpus2:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="-38.4160957,-63.6166725,150km").items(
                    numberOfTweets):
                corpus2.write(tweet.text)
            corpus2.close()
    elif str(paisDos) == "Costa Rica":
        with open("corpus2.txt", "w+", encoding="utf-8") as corpus2:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="9.7489166,-83.7534256,150km").items(
                    numberOfTweets):
                corpus2.write(tweet.text)
            corpus2.close()
    elif str(paisDos) == "Guatemala":
        with open("corpus2.txt", "w+", encoding="utf-8") as corpus2:
            for tweet in tweepy.Cursor(api.search, keyword, geocode="15.7834711,-90.2307587,150km").items(
                    numberOfTweets):
                corpus2.write(tweet.text)
            corpus2.close()
    processData(keyword)


# Funcion que procesa los datos recopilados para sacar las palabras de contexto y alimentarselas a la red de Word2Vec
def processData(keyword):
    with open("corpus.txt", 'r', encoding="utf-8") as corpus:
        for line in corpus:
            linea = str(line)
            x = line.split()
            if "t.co" in x:
                x.remove("t.co")
            if keyword in x:
                answer = lesk(linea, keyword)
                print(answer)
                if answer is not None:
                    print("Answer:" + str(answer) + "Sense:" + str(answer.definition))
                text = x
                # print(bi_tag.tag(x))
                for i in x:
                    if i in stopwords.words('spanish'):
                        x.remove(i)
                    o = x.index(keyword)
                    if o >= 2 & o < len(x) - 2:
                        #print(x[o - 2:o + 3])
                        lista.append(gensim.utils.simple_preprocess(str(x[o - 2:o + 2])))

    with open("corpus2.txt", 'r', encoding="utf-8") as corpus2:
        print('a')
        for line in corpus2:
            y = line.split()
            if keyword in y:
                for i in y:
                    if i in stopwords.words('spanish'):
                        y.remove(i)
                f = y.index(keyword)
                if f >= 2 & f < len(y) - 2:
                    #print(y[f - 2:f + 3])
                    lista.append(gensim.utils.simple_preprocess(str(y[f - 2:f + 2])))




# Algoritmo para WSD(Word Sense Disambiguation)
def lesk(context_sentence, ambiguous_word, pos=None, stem=True, hyperhypo=True):
    max_overlaps = 0;
    lesk_sense = None
    context_sentence = context_sentence.split()
    for ss in wn.synsets(ambiguous_word):
        # If POS is specified.
        #print(ss)
        #print(context_sentence)
        #print(bi_tag.tag(context_sentence))
        if pos and ss.pos is not pos:
            continue

        lesk_dictionary = []

        # Includes definition.
        lesk_dictionary += str(ss.definition).split()
        # Includes lemma_names.
        lesk_dictionary += str(ss.lemma_names).split()

        # Optional: includes lemma_names of hypernyms and hyponyms.
        #if hyperhypo == True:
            #lesk_dictionary += list(chain(*[i.lemma_names for i in ss.hypernyms() + ss.hyponyms()]))

        if stem == True:  # Matching exact words causes sparsity, so lets match stems.
            lesk_dictionary = [ps.stem(i) for i in lesk_dictionary]
            context_sentence = [ps.stem(i) for i in context_sentence]

        overlaps = set(lesk_dictionary).intersection(context_sentence)

        if len(overlaps) > max_overlaps:
            lesk_sense = ss
            max_overlaps = len(overlaps)
    return lesk_sense




# Cuenta de desarrollador que se emplea para sacar la información de Twitter
consumer_key = 'foi4XYdbbbCs0YiYvpTRrx9wE'
consumer_secret = 'CaU79UUkSYzLMTWxJKV5aoBDUm4tFf3BSZjED17z2aSDJBGZIj'
access_token = '1120708300746760194-rHuXz4UqtEeMjIO2wZNZrTMxc4seKI'
access_token_secret = 'r5tbjrLtxrIoeGIQdA7gSczWCuqrzxCUynDsAZZfY5fOT'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)


root = Tk()
label1 = Label(root, text="Homónimo")
E1 = Entry(root, bd=5)
E2 = Entry(root, bd=5)
E3 = Entry(root, bd=5)
E4 = Entry(root, bd=5)

label2 = Label(root, text="Cantidad de la muestra")
label3 = Label(root, text="Zona hispanohablante primaria")
label4 = Label(root, text="Zona hispanohablante secundaria")

lista = []
lista2 = []

ps = PorterStemmer()

# Read the corpus into a list,
# each entry in the list is one sentence.
cess_sents = cess.tagged_sents()

# Train the unigram tagger
uni_tag = ut(cess_sents)

sentence = "Hola , esta foo bar ."

# Tagger reads a list of tokens.
uni_tag.tag(sentence.split(" "))

# Split corpus into training and testing set.
train = int(len(cess_sents)*90/100) # 90%

# Train a bigram tagger with only training data.
bi_tag = bt(cess_sents[:train])

# Evaluates on testing data remaining 10%
bi_tag.evaluate(cess_sents[train+1:])

# Using the tagger.
#bi_tag.tag(sentence.split(" "))


submit = Button(root, text="Someter", command=getData)
# Read the corpus into a list,
# each entry in the list is one sentence.
# text = LazyCorpusLoader('cookbook', TaggedCorpusReader, ["corpus.txt", "corpus2.txt"])
# texto = text.tagged_sents()
#
# # Train the unigram tagger
# uni_tag = ut(texto, train="1")
#
# sentence = "Hola , esta foo bar ."
#
#
# uni_tag.tag(sentence.split(" "))
#
# # Split corpus into training and testing set.
# train = int(len(texto) * 90 / 100)  # 90%
#
# # Train a bigram tagger with only training data.
# bi_tag = bt(texto[:train])
#
# # Evaluates on testing data remaining 10%
# bi_tag.evaluate(texto[train + 1:])
#
# # Using the tagger.
# #bi_tag.tag(sentence.split(" "))
#path = os.path.expanduser('~\\nltk_data')
#if not os.path.exists(path):
  #  os.mkdir(path)
#print(path in nltk.data.path)


root.wm_title('Mutación en Sede')
label1.pack()
E1.pack()

label2.pack()
E2.pack()
label3.pack()
E3.pack()
label4.pack()
E4.pack()

submit.pack(side=BOTTOM)
root.mainloop()
