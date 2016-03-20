# This program is part of prosaic.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
from itertools import takewhile, dropwhile
from functools import partial, lru_cache
import re
import sys
from os.path import join, expanduser, exists
import nltk
from prosaic.util import match, invert, first, compose, second, some, is_empty, last, find_first
from prosaic.cfg import LANGUAGE

# We have to pause our imports, here, to do some NLTK prep. We can't import
# certain things until we've downloaded raw corpora and other data, so we do so
# here:
NLTK_DATA_PATH = join(expanduser('~'), 'nltk_data')
NLTK_DATA = ['punkt',
             'maxent_ne_chunker',
             'cmudict',
             'words',
             'maxent_treebank_pos_tagger',]

if not exists(NLTK_DATA_PATH):
    for datum in NLTK_DATA:
        nltk.download(datum)

import nltk.chunk as chunk
from nltk.corpus import cmudict

DIVIDER_TAG = ':' # nltk uses this to tag for ; and :

# Set up some state that we'll use in the functions throughout this file:
# TODO consider making a class that has modular stemmer/tokenizer
if LANGUAGE == "english":
    from nltk.stem.snowball import EnglishStemmer
    stemmer = EnglishStemmer()
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    cmudict_dict = cmudict.dict()

    # Some useful regexes:
    vowel_re = re.compile("[aeiouAEIOU]")
    vowel_phoneme_re = re.compile("AA|AE|AH|AO|AW|AY|EH|EY|ER|IH|IY|OW|OY|UH|UW")
    consonant_phoneme_re = re.compile("^(?:B|D|G|JH|L|N|P|S|T|V|Y|ZH|CH|DH|F|HH|K|M|NG|R|SH|TH|W|Z)")
elif LANGUAGE == "german":
    from nltk.stem.snowball import GermanStemmer
    stemmer = GermanStemmer()
    tokenizer = nltk.data.load("tokenizers/punkt/german.pickle")
    cmudict_dict = cmudict.dict()
    
    vowel_re = re.compile("[aeiouAEIOUÖÜÄYy]")
    vowel_phoneme_re = re.compile("AA|AI|AO|AU|AY|EE|EI|EU|EY|IE|II|OO|OU|OY|UI|UO|UU|ÄU|AE|OE|UE")
    consonant_phoneme_re = re.compile("^(?:B|D|G|JH|L|N|P|S|T|V|Y|ZH|CH|DH|F|HH|K|M|NG|R|SH|TH|W|V|Z|SCH)")

# Helper predicates:
is_vowel = partial(match, vowel_re)
is_vowel_phoneme = partial(match, vowel_phoneme_re)
is_consonant_phoneme = partial(match, consonant_phoneme_re)

def word_to_phonemes(word):
    result = cmudict_dict.get(word.lower(), None)
    if LANGUAGE == "german":
        print("This most likely generates rubbish. There is no german phoneme database implemented")
    if result is None:
        # TODO I don't really like this. Should at least return None.
        return []
    else:
        return first(result)

sentences = lambda raw_text: tokenizer.tokenize(raw_text)

@lru_cache(maxsize=256)
def stem_word(word):
    return stemmer.stem(word)

@lru_cache(maxsize=2056)
def tag(sentence_string):
    tokenized_words = nltk.word_tokenize(sentence_string)
    return nltk.pos_tag(tokenized_words)

word_tag_re = re.compile("^[A-Z]+$")
@lru_cache(maxsize=2056)
def words(sentence):
    tagged_sentence = tag(sentence)
    tagged_words = filter(lambda tu: match(word_tag_re, second(tu)), tagged_sentence)
    ws = map(first, tagged_words)
    return list(ws)

def stem_sentence(sentence):
    stemmed = map(stem_word, words(sentence))
    return list(stemmed)

is_divider = lambda tu: DIVIDER_TAG == second(tu)

def split_multiclause(sentence, tagged_sentence):
    # extract the text the divider tag represents
    divider = first(find_first(is_divider, tagged_sentence))
    if divider is not None:
        first_clause = sentence[0:sentence.index(divider)].rstrip()
        second_clause = sentence[sentence.index(divider)+1:].lstrip()
        return [first_clause, second_clause]
    else:
        return [sentence]

def expand_multiclauses(sentences):
    # TODO consider itertools
    split = []
    for sentence in sentences:
        tagged_sentence = tag(sentence)
        if not is_empty(tagged_sentence):
            split += split_multiclause(sentence, tagged_sentence)
    return split

# TODO Ideally we'd store the original sentence along with the tagged version,
# but that gets slightly hard with multiclause expansion.
punctuation_regex = re.compile("^[^a-zA-Z0-9]")

@lru_cache(maxsize=256)
def match_punctuation(string):
    return match(punctuation_regex, string)

def count_syllables_in_word(word):
    phonemes = word_to_phonemes(word)
    if phonemes:
        # count vowel syllables:
        vowel_things = filter(is_vowel_phoneme, phonemes)
    else:
        # raw vowel counting:
        vowel_things = filter(is_vowel, list(word))

    return len(list(vowel_things))

def count_syllables(sentence):
    syllable_counts = map(count_syllables_in_word, words(sentence))
    return sum(syllable_counts)

alpha_tag = re.compile("^[a-zA-Z]")
is_alpha_tag = partial(match, alpha_tag)

def rhyme_sound(sentence):
    tagged_sentence = tag(sentence)
    without_punctuation = filter(compose(is_alpha_tag, second), tagged_sentence)
    ws = list(map(first, without_punctuation))

    if is_empty(ws):
        return None

    last_word = last(ws)
    phonemes = word_to_phonemes(last_word)

    if is_empty(phonemes):
        return None

    return "".join(phonemes[-3:])

if LANGUAGE == "english":
    consonant_re = re.compile("(SH|CH|TH|B|D|G|L|N|P|S|T|V|Y|F|K|M|NG|R|W|Z)")
elif LANGUAGE == "german":
    consonant_re = re.compile("(B|C|D|G|H|L|M|N|P|Q|R|S|T|V|X|F|K|W|Z)")

def has_alliteration(sentence):
    ws = words(sentence)

    def first_consonant_sound(word):
        phonemes = word_to_phonemes(word)
        if not is_empty(phonemes):
            return find_first(is_consonant_phoneme, phonemes)
        else:
            return first(consonant_re.findall(word.upper()))

    first_consonant_phonemes = map(first_consonant_sound, ws)
    last_phoneme = None
    for phoneme in first_consonant_phonemes:
        if last_phoneme is None:
            last_phoneme = phoneme
        else:
            if last_phoneme == phoneme:
                return True
            else:
                last_phoneme = phoneme
    return False

# Function taken from: https://github.com/tradloff/haiku
def count_syllables_in_word_german (word):
    VOWELS_DE = "aeiouyäöü"
#      Vokale der deutschen Sprache, einschliesslich "y" und Umlaute.
       
    ONE_SYLLABLE_COMBINATIONS = (
            "aa", "ai", "ao", "au", "ay",
            "ee", "ei", "eu", "ey",
            "ie", "ii",
            "oo", "ou", "oy",
            "ui", "uo", "uu",
            "\xe4u",
            "ae", "oe", "ue"
    )
    SYL_DIGITS_REGEXP = re.compile(r"\d+")
    SYL_QU_REGEXP = re.compile(r"qu[aeiouäöüy]")
    SYL_Y_REGEXP = re.compile(r"y[aeiouäöü]")
    SYLLABLE_COUNT_EXCEPTIONS = {"Pietät": 3, "McDonald's": 3, "T-Shirt": 2, "orange": 2}
    #Rekursive Funktion, die die Zahl der Silben fuer ein  deutsches Wort zurueckgibt.
    

    # Manchmal gibt libleipzig merere Worte als ein einzelnes zurueck
    if " " in word:
            return sum([countSyllables(w) for w in word.split()])

    # Sonderfall: Bindestrich
    word = word.strip("-")
    if "-" in word:
            return sum([countSyllables(w) for w in word.split("-")])

    def __sylCount(charList):
#            Rekursive Funktion, die die naechste Vokalkette sucht und die Anzahl ihrer Silben bestimmt.

            if charList == []:
                    return 0
            c, v = 0, []
            while c < len(charList) and charList[c] not in VOWELS_DE:
                    c += 1
            while c < len(charList) and charList[c] in VOWELS_DE:
                    v.append(charList[c])
                    c += 1
            # kein Vokal: keine Silbe
            if v == []:
                    return 0
            # ein Vokal: eine Silbe
            elif len(v) == 1:
                    return 1 + __sylCount(charList[c:])
            # zwei Vokale: eine oder zwei Silben
            elif len(v) >= 2:
                    if "".join(v[:2]) in ONE_SYLLABLE_COMBINATIONS:
                            return 1 + __sylCount(charList[c-len(v)+2:])
                    else:
                            return 1 + __sylCount(charList[c-len(v)+1:])

    # Wort in Unicode umwandeln
    try:
            word = word
    except UnicodeEncodeError as u:
            pass

    # Ausnahmen abfragen
    if word in list(SYLLABLE_COUNT_EXCEPTIONS.keys()):
            return SYLLABLE_COUNT_EXCEPTIONS[word]

    # Sonderzeichen eliminieren
    if not word.isalnum():
            word = "".join([w for w in word if w.isalnum() or w in "'`-"]).rstrip("-")

    # Sonderfall: Abkuerzung
    if word.isupper():
            return len(word) + word.count("Y") * 2 # "Ypsilon" hat zwei Silben mehr
    if word[:-1].isupper() and word[-1] == "s": # Plural-Abkuerzung
            return len(word) - 1 + word.count("Y") * 2

    word = word.lower()

    # Sonderfall: Ziffern am Wortanfang (zB "1920er")
    m = SYL_DIGITS_REGEXP.match(word)
    if m != None:
            return m.end() + countSyllables(word[m.end():])

    # Sonderfall: "y<vokal>" ist eine Silbe -> "y" abschneiden
    if SYL_Y_REGEXP.match(word):
            word = word[1:]

    # Sonderfall: "qu<vokal>" ist eine Silbe -> durch "qu" ersetzen
    word = SYL_QU_REGEXP.sub("qu", word)

    return __sylCount(list(word))
