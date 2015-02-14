#/bin/sh

# dacapo
# 1～10 : not overlap
python petrims.py -c -t de test_dacapo/result_1.log 2 2> test_dacapo/error_1.log
python petrims.py -c -t de test_dacapo/result_2.log 2 2> test_dacapo/error_2.log
python petrims.py -c -t de test_dacapo/result_3.log 2 2> test_dacapo/error_3.log
python petrims.py -c -t de test_dacapo/result_4.log 2 2> test_dacapo/error_4.log
python petrims.py -c -t de test_dacapo/result_5.log 2 2> test_dacapo/error_5.log
python petrims.py -c -t de test_dacapo/result_6.log 2 2> test_dacapo/error_6.log
python petrims.py -c -t de test_dacapo/result_7.log 2 2> test_dacapo/error_7.log
python petrims.py -c -t de test_dacapo/result_8.log 2 2> test_dacapo/error_8.log
python petrims.py -c -t de test_dacapo/result_9.log 2 2> test_dacapo/error_9.log
python petrims.py -c -t de test_dacapo/result_10.log 2 2> test_dacapo/error_10.log
# 11～20 : overlap
python petrims.py -t de test_dacapo/result_11.log 2> test_dacapo/error_11.log
python petrims.py -t de test_dacapo/result_12.log 2> test_dacapo/error_12.log
python petrims.py -t de test_dacapo/result_13.log 2> test_dacapo/error_13.log
python petrims.py -t de test_dacapo/result_14.log 2> test_dacapo/error_14.log
python petrims.py -t de test_dacapo/result_15.log 2> test_dacapo/error_15.log
python petrims.py -t de test_dacapo/result_16.log 2> test_dacapo/error_16.log
python petrims.py -t de test_dacapo/result_17.log 2> test_dacapo/error_17.log
python petrims.py -t de test_dacapo/result_18.log 2> test_dacapo/error_18.log
python petrims.py -t de test_dacapo/result_19.log 2> test_dacapo/error_19.log
python petrims.py -t de test_dacapo/result_20.log 2> test_dacapo/error_20.log
