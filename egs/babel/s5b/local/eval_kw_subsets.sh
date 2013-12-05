KWSEval -e ecf.xml -r rttm -t keyword_outvocab.xml  -s kwslist.xml -c -o -b -d -f ./kws/outvocab
KWSEval -e ecf.xml -r rttm -t keyword_invocab.xml  -s kwslist.xml -c -o -b -d -f ./kws/invocab
KWSEval -e ecf.xml -r rttm -t kws.xml  -s kwslist.xml -c -o -b -d -f ./kws/fullvocab

