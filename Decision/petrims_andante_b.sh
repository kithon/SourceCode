#/bin/sh

# andante
# makeup random extreme decision tree
python petrims.py -c -p test_andante/result_1.log -t b test_andante/result_1.log 2 2> test_andante/error_1.log
python petrims.py -c -p test_andante/result_2.log -t b test_andante/result_2.log 2 2> test_andante/error_2.log
python petrims.py -c -p test_andante/result_3.log -t b test_andante/result_3.log 2 2> test_andante/error_3.log
python petrims.py -c -p test_andante/result_4.log -t b test_andante/result_4.log 2 2> test_andante/error_4.log
python petrims.py -c -p test_andante/result_5.log -t b test_andante/result_5.log 2 2> test_andante/error_5.log
python petrims.py -c -p test_andante/result_6.log -t b test_andante/result_6.log 2 2> test_andante/error_6.log
python petrims.py -c -p test_andante/result_7.log -t b test_andante/result_7.log 2 2> test_andante/error_7.log
python petrims.py -c -p test_andante/result_8.log -t b test_andante/result_8.log 2 2> test_andante/error_8.log
python petrims.py -c -p test_andante/result_9.log -t b test_andante/result_9.log 2 2> test_andante/error_9.log
python petrims.py -c -p test_andante/result_10.log -t b test_andante/result_10.log 2 2> test_andante/error_10.log