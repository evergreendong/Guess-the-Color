# -*- coding: utf-8 -*-

from collections import defaultdict
from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet as wn
import string
import nltk

nltk.download('brown')
nltk.download('wordnet')
sentences = brown.tagged_sents()
lmt = WordNetLemmatizer()

# all of the color names, taken from webcolors packages
color_names = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', \
        'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', \
        'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', \
        'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen',
        'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', \
        'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', \
        'dimgray', 'dimgrey', 'dodgerblue','firebrick', 'floralwhite', 'forestgreen', 'fuchsia', \
        'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green', 'greenyellow', \
        'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', \
        'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', \
        'lightgray', 'lightgrey', 'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', \
        'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', \
        'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', \
        'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', \
        'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', \
        'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', \
        'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', \
        'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', \
        'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', \
        'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', \
        'whitesmoke', 'yellow', 'yellowgreen']

def process(s):
    # preprocess all the words
    result = []
    for w,t in s:
        if w in string.punctuation:
            continue
        if w in ["``","''"]:
            continue
        v = lmt.lemmatize( w ).lower()
        result.append( (v,t) )
    return result

def get_2gram(s):
    N_word = len(s)
    i = N_word - 2
    while i >= 0:
        p1 = s[i]
        p2 = s[i+1]
        yield (p1,p2)
        i -= 1

def predict(c,w):
    # predict the true color give partial color character c and the noun w
    def most_color(color_freq, c):
        # find most frequence color start with c inside color_freq
        # color_freq is a dict where keys are color and values are #times appears
        tmp = [(v,k) for k,v in color_freq.items()]
        tmp.sort(reverse = True)
        for v,k in tmp:
            if k.startswith(c):
                return k
        # no color in this dict start with character c
        return None

    if w not in memory:
        return most_color(memory['color'], c)

    clr = most_color(memory[w], c)
    if clr is None:
        return most_color(memory['color'], c)
    else:
        return clr


def main(infile, outfile):
    # read file content, predict and write answer back to file
    fh = open(infile, 'r')
    content = fh.readlines()
    fh.close()

    fh = open(outfile, 'w')
    for line in content:
        c,w = line.strip().split(' ')
        full_color = predict(c,w)
        print(full_color, w)
        fh.write('{} {}\n'.format(full_color, w))

    fh.close()
    return

# find all 2 gram where first word describe a color and second word is Noun.
memory = {}
memory['color'] = defaultdict(int)

for s in sentences:
    for (w1,t1),(w2,t2) in get_2gram(process(s)):
        if w1 in color_names:
            if not t2.startswith('NN'):
                continue

            if w2 not in memory:
                memory[w2] = defaultdict(int)
            memory[w2][w1] += 1
            memory['color'][w1] += 1


if __name__ == '__main__':
    import sys
    if (len(sys.argv) > 2):
        infile = sys.argv[1]
        outfile = sys.argv[2]
        main(infile, outfile)
    else:
        print("Not enough arguments")