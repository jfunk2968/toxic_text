#!/bin/bash
python pytorch_linear.py toxic 300 single > toxic_single_log.txt
python pytorch_linear.py severe_toxic 300 single > severe_toxic_single_log.txt
python pytorch_linear.py obscene 300 single > obscene_single_log.txt
python pytorch_linear.py insult 300 single > insult_single_log.txt
python pytorch_linear.py threat 300 single > threat_single_log.txt
python pytorch_linear.py identity_hate 300 single > identity_hate_single_log.txt