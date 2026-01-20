#! /usr/bin/env python
import sys

word          = ''
MINCOUNT      = 3
MAXCOUNT      = 800

word2count = {}
maxed      = set()

for line in sys.stdin:
    line = line.strip()

    (word, count) = line.split('\t', 1)
    if (word in maxed):
        continue
    count = int(count)

    try:
        word2count[word] = word2count[word]+count
        if (word2count[word] >= MAXCOUNT):
            maxed.add(word)
    except:
        word2count[word] = count

for word,count in word2count.items():
    if (count > MINCOUNT):
        print ('%s\t%s'% ( word, count ))
