import os
import re
import json
import operator
import itertools as it
from nltk.tokenize import sent_tokenize
import subprocess
import wikipedia


def wiki_extract(pagelist):
    """

    :param pagelist:
    :return:
    """
    with open('unilex.json') as json_data:
        lexicon = json.load(json_data)
    count = 0
    sentences = []
    sentence_list = []
    for l, p in enumerate(pagelist):
        print(l, "sentences from '{}'".format(p).upper())
        try:
            sentence_list = sent_tokenize(wikipedia.page(p).content)
        except Exception as e:
            print(e)
        sentence_list = [s for s in sentence_list if '\n' not in s]
        sentence_list = [s.replace('"', '') for s in sentence_list]
        for sentence in sentence_list:
            clean_list = re.sub(r'[\.\,\?\!\"]', '', sentence.lower()).split()
            print(clean_list)
            if set(clean_list).issubset(lexicon) and len(clean_list) in range(4, 15):
                count += 1
                sentences.append(sentence)
    print("THE NUMBER OF SENTENCES FOR SELECTION IS:", sentences)
    with open('corpus.json', 'w') as outfile:
        json.dump(sentences, outfile)
    return(sentences)



### open json file with all sentences ###
with open('corpus.json') as json_data_2:
    sentence_init = json.load(json_data_2)
    sentences=[]
    for x in sentence_init:
        if x not in sentences:
            sentences.append(x)
    print(len(sentences))


### create utts.data file for festival ###
def make_sentence_dict(sentence_list):
    try:
        os.remove("sequences.txt")
        os.remove("utts.data")
    except OSError:
        pass
    count2 = 0
    sentences_dict = {}
    with open("utts.data", 'a') as output_file:
        for n, sentence in enumerate(sentence_list):
            sentences_dict["*/irnbros_"+str(count2).zfill(5)] = sentence
            count2 += 1
            string = ""
            string += "( irnbros_"+str(count2).zfill(5)+' "'
            string += sentence
            string += '" )\n'
            output_file.write(string)
    return sentences_dict

def get_diphones_festival():
    sentence_diphone_dict = {}
    ### get phone sequences through festival and convert to diphone sequences ###
    count_irn=0
    print("open festival...")
    cmd = ['./festival.sh', '--arg', 'value']  # creates utts.mlf
    subprocess.Popen(cmd, stdout=True).wait()

    with open("utts.mlf", 'r') as f, open("sequences.txt", 'a') as sequences:
        for key,group in it.groupby(f,lambda line: line.startswith('#!MLF!#')):
            if not key:
                group = list(group)
                x = [s.strip('\n') for s in group]
                x = [s for s in x if '_cl' not in s]
                x = [s for s in x if 'sp' not in s]
                key = "*/irnbros_" + str(count_irn).zfill(5)
                count_irn += 1
                value = x[1:-1]
                diphone_value = ['_'.join(value[i:i + 2]) for i in range(len(value) - 1)]
                sentence_diphone_dict[key] = diphone_value
                sequences.write(str(key)+str(diphone_value)+'\n')
    return sentence_diphone_dict

def get_diphone_wishlist(sentence_diphone_dict):
    """
    get a wishlist of diphones in dictionary with diphone as key and number of occurrences as value
    :param sentence_diphone_dict: a dictionary with sentence key and diphones in that sentence
    :return: diphone_wishlist
    """
    diphone_wishlist = {} # dictionary with diphones
    for k, v in sentence_diphone_dict.items():
        for type in v:  # making the wishlist for diphones
            if type in diphone_wishlist:
                diphone_wishlist[type] += 1
            else:
                diphone_wishlist[type] = 1
    return diphone_wishlist

try:
    os.remove("wishlist.txt")
    os.remove("wishlist_diphones.txt")
    os.remove("rank_dict.json")
    os.remove("rank_dict_rev.json")
    os.remove("figure_1.pdf")
    os.remove("figure_2.pdf")
except OSError:
    pass

triphones_together = []



def script_generator(sentences_dict, sentence_diphone_dict, diphone_wishlist):
    diphone_inventory={}
    dict_final={}
    try:
        os.remove("script_2.txt")
    except OSError:
        pass
    with open("script_2.txt","a") as script:
        for n in range(55):
            score_dict = {}
            discores = {}
            for k, v in sentence_diphone_dict.items():
                discores[k] = []
                score = 0
                for i, diphone in enumerate(v):
                    try:
                        score += 1 / diphone_wishlist[diphone]
                        individual_score = 1 / diphone_wishlist[diphone]
                    except:
                        pass
                    score2 = score / len(v)
                    score_dict[k] = score2
            best = max(score_dict.items(), key=operator.itemgetter(1))[0]  # finding max value in dict
            best_sentence = sentences_dict[best]  # actual sentence
            print(n+1, best_sentence)
            script.write(best_sentence+" ("+str(score_dict[best])+")"+"\n"+str(discores[best])+"\n")  # write to file script
            print('phones left: ', len(diphone_wishlist))
            for diphone in sentence_diphone_dict[best]:
                try:
                    diphone_wishlist.pop(diphone)
                except:
                    pass
            dict_final[best] = best_sentence
            sentence_diphone_dict.pop(best)
    return dict_final



if __name__ == "__main__":
    pagelist = wikipedia.search("cheese", results=5, suggestion=False)
    print(pagelist)
    sentences = wiki_extract(pagelist)
    sentences_dict = make_sentence_dict(sentences)
    sentence_diphone_dict = get_diphones_festival()
    diphone_wishlist = get_diphone_wishlist(sentence_diphone_dict)
    dict_final = script_generator(sentences_dict, sentence_diphone_dict, diphone_wishlist)

    try:
        os.remove("utts.data")
    except OSError:
        pass

    with open("utts1.data", 'w') as final_inventory:
        for k, v in dict_final.items():
            string = v + '\n'
            final_inventory.write(string)

