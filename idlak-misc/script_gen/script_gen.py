"""
This script takes an idlak database structured as specified (github.com/idlak/idlak-resources) and parses
all of its sentences through the front-end of Idlak. From the resulting xml-file, phone-information per sentence
is converted into diphone information and stored with a sentence-id.

A diphone wishlist is made which contains every diphone present in the database and how often it occurs.
This information is used to pick sentences iteratively that add the most, new uncommon diphones.
"""

# TODO: make cleaner script
# TODO: make python2 compatible
# TODO: limit sentence length for script


import operator
import subprocess
from lxml import etree
import logging
import argparse
import re
import os
import json


logging.basicConfig(filename='example.log',level=logging.DEBUG)


parser = argparse.ArgumentParser(description='Specify location for temporary file storage and of the Wikipedia dump'
                                             'to generate an xml file in Idlak structure ')
parser.add_argument('idlak_path', nargs='+', help="specify idlak path")
parser.add_argument('language', nargs='+', help="specify language code you're working with")
parser.add_argument('dialect', nargs='+', help="specify dialect code of language you're working with")
parser.add_argument('database_path', nargs='+', help="specify database file path")
args = parser.parse_args()


def get_txp_input(idlak_database):
    """
    Takes idlak database returns file path to the converted database ready for the ./idlaktxp
    :param idlak_database: xml database in idlak format
    :return: path of idlaktxp-ready file
    """
    tree = etree.parse(idlak_database)
    string = ''
    for child in tree.getroot():
        string += re.sub(r'[\n\r]', ' ', re.sub(r'[<>]', '', child[0].text))
    # new_out = ''
    # for line in string.splitlines():
    #     if len(line.split()) > 5:
    #         if not re.match(r'^\s*$', line):
    #             new_out += line + '\n'
    final_string = '<parent>\n' + string + '</parent>'
    with open('idlak_input.xml', 'w') as outfile:
        outfile.write(final_string)
    return os.path.abspath('idlak_input.xml')


def get_txp_output(txp_input, idlak_location, language, dialect):
    """
    takes location of idlak and xml input file as well as a language and dialect code to generate an output xml with
    the phonetic specification of all the text in the input file.
    :param txp_input: path of xml input file
    :param idlak_location: path of idlak installation
    :param language: language code
    :param dialect: accent/dialect code
    :return: path of generated output
    """
    current_loc = os.getcwd()
    cmd1 = 'cd {}/src/idlaktxpbin'.format(idlak_location)
    cmd2 = "./idlaktxp --tpdb=../../idlak-data --general-lang={} --general-acc={} {} {}/output.xml".format(
        language, dialect, txp_input, current_loc)
    subprocess.call("{}; {}".format(cmd1, cmd2), shell=True)

    return os.getcwd() + '/output.xml'


def sentence_dict_gen(xml_file):
    """
    Takes an idlaktxp output file and retrieves utterance-id and the utterance. A dictionary with these two elements
    is returned.
    :param xml_file: path of idlaktxp output file
    :return: dict with utterance-id as key and utterance as value
    """
    tree = etree.parse(xml_file)
    sentences = {}
    for child in tree.getroot():
        if 5 <= len([element for phrase in child.iter("spt") for element in phrase.iter("tk")]) <= 16:
            string = ''
            for phrase in child.iter("spt"):
                len(phrase.getchildren())
                for element in phrase.iter("tk"):
                    try:
                        string += element.text + ' '
                    except TypeError as e:
                        print(e)
                    #  logging.info(string)
            sentences[child.get("uttid").zfill(8)] = string
    return sentences


def phone_dict_gen(xml_file):
    """
    Takes an idlaktxp output file and retrieves utterance-id and the utterance phones. These phones are converted into
    diphones. A dictionary with these two elements is returned.
    :param xml_file: path of idlaktxp output file
    :return: dict with utterance-id as key and utterance diphones as value
    """
    tree = etree.parse(xml_file)
    sent_diphones = {}
    for child in tree.getroot():
        if 5 <= len([element for phrase in child.iter("spt") for element in phrase.iter("tk")]) <= 16:
            phones = ['silbb']
            for phrase in child.iter("spt"):
                for element in phrase.iter("tk"):
                    phones.extend(element.get("pron").split())
                phones.append('silb')
            phones[-1] = 'silbb'
            diphones = ['_'.join(phones[i:i + 2]) for i in range(len(phones) - 1)]
            sent_diphones[child.get("uttid").zfill(8)] = diphones
    return sent_diphones


def diphone_wishlist_gen(diphone_dict):
    """
    Take a dictionary with sentences and diphones and construct a diphone wishlist
    :param diphone_dict: dict with diphones of each sentence
    :return: dict with diphone and frequency count
    """
    diphone_wishlist = {} # dictionary with diphones
    for k, v in diphone_dict.items():
        for type in v:  # making the wishlist for diphones
            if type in diphone_wishlist:
                diphone_wishlist[type] += 1
            else:
                diphone_wishlist[type] = 1
    with open('wishlist_out.txt', 'w') as out:
        out.write(json.dumps(sorted(diphone_wishlist.items(), key=lambda x: x[1]), indent=4))
    return diphone_wishlist


def script_generator(sentences_dict, sentence_diphone_dict, diphone_wishlist, number_of_sentence):
    """
    Generates a script with the most phonetically rich sentences.
    Rarer diphones are prioritised over more common ones.
    :param sentences_dict: Sentence keys with sentences as values
    :param sentence_diphone_dict: Sentence keys with diphones of each sentence as values
    :param diphone_wishlist: Diphone wishlist with diphone as key and number of time it occurs in the sentences as value
    :param number_of_sentence: The number of sentences the script should contain
    :return: Dictionary with 'best' sentences for the script
    """
    dict_final = {}
    for n in range(number_of_sentence):  # need to ensure the script doesn't fail when this range is higher
                                        # than sentence_diphone_dict.items()
        score_dict = {}
        discores ={}
        for k, v in sentence_diphone_dict.items():
            discores[k]=[]
            score = 0
            for i, diphone in enumerate(v):
                try:
                    score += 1 / diphone_wishlist[diphone]
                    individual_score = 1 / diphone_wishlist[diphone]
                    discores[k].extend([diphone, individual_score])
                except:
                    pass
                score2 = score / len(v)
                score_dict[k] = score2
        best = max(score_dict.items(), key=operator.itemgetter(1))[0]  # finding max value in dict
        best_sentence = sentences_dict[best]  # actual sentence
        logging.info(best_sentence+" ("+str(score_dict[best])+")"+"\n"+str(discores[best])+"\n")  # write score to log
        # print('phones left: ', len(diphone_wishlist))
        for diphone in sentence_diphone_dict[best]:
            try:
                diphone_wishlist.pop(diphone)
            except:
                pass
        dict_final[str(n).zfill(5)] = best_sentence
        sentence_diphone_dict.pop(best)
    return dict_final


if __name__ == "__main__":
    txp_input = get_txp_input('database-nl.xml')
    txp_output = get_txp_output(txp_input, args.idlak_path[0], args.language[0], args.dialect[0])

    sentences = sentence_dict_gen(txp_output)
    sent_phones = phone_dict_gen(txp_output)
    wishlist = diphone_wishlist_gen(sent_phones)
    script = script_generator(sentences, sent_phones, wishlist, 30)

    subprocess.run(['rm', '-R', 'script.txt'])

    with open("script.txt", 'w') as final_inventory:
        for k, v in script.items():
            string = k + ': ' + v + '\n'
            final_inventory.write(string)
