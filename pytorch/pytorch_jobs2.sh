#!/bin/bash
python pytorch.py toxic 300 single > single/toxic_log.txt
python pytorch.py severe_toxic 300 single > single/severe_toxic_log.txt
python pytorch.py obscene 300 single > single/obscene_log.txt
python pytorch.py insult 300 single > single/insult_log.txt
python pytorch.py threat 300 single > single/threat_log.txt
python pytorch.py identity_hate 300 single > single/identity_hate_log.txt