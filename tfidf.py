#Mengimport Library untuk membuka file csv
import csv
import math
import pandas as pd
#Mengimport Library Sastrawi untuk Stemming dan Stopword Removal
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
#Membuat Stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()
# Membuat stopword remover
factoryy = StopWordRemoverFactory()
stpwrd = factoryy.create_stop_word_remover()
#Membuka File untuk ditulis
csvFile = open('Diagnosis.csv', 'a')
csvWriter = csv.writer(csvFile)
bowA = []
# Membuat fungsi untuk enghitung TF IDF dari data
def computeReviewTFDict(review):
    reviewTFDict = {}
    for word in review:
        if word in reviewTFDict:
            reviewTFDict[word] += 1
        else:
            reviewTFDict[word] = 1           
    for word in reviewTFDict:
        reviewTFDict[word] = round((reviewTFDict[word] / len(review)),4)
    return reviewTFDict
#Membuat fungsi penghitung kata
def computeCountDict(tfDict):
        countDict = {}
        for review in tfDict:
            for word in review:
                if word in countDict:
                    countDict[word] += 1
                else:
                    countDict[word] = 1
        return countDict
# Membuat fungsi untuk menghitung IDF setiap kata
def computeIDFDict(countDict, data):
    idfDict = {}
    for word in countDict:
        idfDict[word] = math.log(len(data) / countDict[word])
    return idfDict
# Membuat fungsi untuk menghitung TF-IDF dari setiap kata dari setiap dokumen
def computeReviewTFIDFDict(reviewTFDict, idfDict):
    reviewTFIDFDict = {}
    for word in reviewTFDict:
        reviewTFIDFDict[word] = round(reviewTFDict[word] * idfDict[word],4)
    return reviewTFIDFDict
# Fungsi vektorisasi tfidf
def computeTFIDFVector(review):
    tfidfVector = [0.0] * len(wordDict)
    
    for i, word in enumerate(wordDict):
        if word in review:
            tfidfVector[i] = review[word]
    return tfidfVector

#Membaca file hasil mining tweet
with open('pengpol.csv', 'r') as pegang:
    csvReader = csv.DictReader(pegang)
    for row in csvReader:
        # Stemming menggunakan library Sastrawi
        output = stemmer.stem(row['Gejala'])
        # Stop word removal
        stp = stpwrd.remove(output)
        # Tokenization
        result = stp.split()
        # Mengeluarkan hasil di terminal
        bowA.append(result)
    wordDict = {"demam"}
    tfdicts = {}
    # Menghitung tf dari setiap kata dalam setiap dokumen
    for i in range(len(bowA)):
        tfdicts[i] = computeReviewTFDict(bowA[i])
    # Menghitung jumlah setiap kata dalam seluruh dokumen
    countDict = computeCountDict(bowA)
    # Menghitung IDF setiap kata dalam dokumen
    idfs = computeIDFDict(countDict, bowA)
    #Menghitung TF-IDF dari setiap kata dalam setiap dokumen
    tfidfDict = [computeReviewTFIDFDict(tfdicts[i], idfs) for i in range(len(tfdicts))]
    # Mengeluarkan hasil tfidf
    print(tfidfDict)
    # Menyimpan hasil tfidf per kata ke dokumen laporan
    csvWriter.writerow(result)
    # Menyimpan seluruh kata ke dalam wordDict
    wordDict = sorted(countDict.keys())
    print(len(wordDict))
    # Vektorisasi namun masih gagal
    tfidfVector = [computeTFIDFVector(bowA) for review in tfidfDict[i]]

