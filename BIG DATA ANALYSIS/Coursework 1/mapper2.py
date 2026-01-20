#! /usr/bin/env python3
import sys
import re
import string

nomber = re.compile("[0-9]")

for line in sys.stdin:
    words = line.translate(str.maketrans('','',string.punctuation))
    words = words.strip().lower()
    for word in words.split():
        if (re.findall(nomber, word)):
            word = "NOMBER"

        print("%s\t%s" % (word,1)) #alternative to emit()

