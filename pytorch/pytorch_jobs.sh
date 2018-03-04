#!/bin/bash
python pytorch.py toxic 300 linear > linear/toxic_log.txt
python pytorch.py severe_toxic 300 linear > linear/severe_toxic_log.txt
python pytorch.py obscene 300 linear > linear/obscene_log.txt
python pytorch.py insult 300 linear > linear/insult_log.txt
python pytorch.py threat 300 linear > linear/threat_log.txt
python pytorch.py identity_hate 300 linear > linear/identity_hate_log.txt