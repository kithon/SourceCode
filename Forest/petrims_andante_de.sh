#/bin/sh

# andante
# 1～10 : not overlap
python petrims.py -c -t de test_andante/result_1.log 2 2> test_andante/error_1.log
python petrims.py -c -t de test_andante/result_2.log 2 2> test_andante/error_2.log
python petrims.py -c -t de test_andante/result_3.log 2 2> test_andante/error_3.log
python petrims.py -c -t de test_andante/result_4.log 2 2> test_andante/error_4.log
python petrims.py -c -t de test_andante/result_5.log 2 2> test_andante/error_5.log
python petrims.py -c -t de test_andante/result_6.log 2 2> test_andante/error_6.log
python petrims.py -c -t de test_andante/result_7.log 2 2> test_andante/error_7.log
python petrims.py -c -t de test_andante/result_8.log 2 2> test_andante/error_8.log
python petrims.py -c -t de test_andante/result_9.log 2 2> test_andante/error_9.log
python petrims.py -c -t de test_andante/result_10.log 2 2> test_andante/error_10.log
# 11～20 : overlap
python petrims.py -t de test_andante/result_11.log 2> test_andante/error_11.log
python petrims.py -t de test_andante/result_12.log 2> test_andante/error_12.log
python petrims.py -t de test_andante/result_13.log 2> test_andante/error_13.log
python petrims.py -t de test_andante/result_14.log 2> test_andante/error_14.log
python petrims.py -t de test_andante/result_15.log 2> test_andante/error_15.log
python petrims.py -t de test_andante/result_16.log 2> test_andante/error_16.log
python petrims.py -t de test_andante/result_17.log 2> test_andante/error_17.log
python petrims.py -t de test_andante/result_18.log 2> test_andante/error_18.log
python petrims.py -t de test_andante/result_19.log 2> test_andante/error_19.log
python petrims.py -t de test_andante/result_20.log 2> test_andante/error_20.log
