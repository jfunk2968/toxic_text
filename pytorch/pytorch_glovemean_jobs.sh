#!/bin/bash
python pytorch_glove.py toxic 300  > glove_mean/toxic_log.txt
python pytorch_glove.py severe_toxic 300  > glove_mean/severe_toxic_log.txt
python pytorch_glove.py obscene 300  > glove_mean/obscene_log.txt
python pytorch_glove.py insult 300  > glove_mean/insult_log.txt
python pytorch_glove.py threat 300  > glove_mean/threat_log.txt
python pytorch_glove.py identity_hate 300  > glove_mean/identity_hate_log.txt