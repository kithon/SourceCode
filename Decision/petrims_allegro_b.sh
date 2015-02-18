#/bin/sh

# allegro
# makeup random extreme decision tree
python petrims.py -c -p test_allegro/result_1.log -t b test_allegro/result_1.log 2 2> test_allegro/error_1.log
python petrims.py -c -p test_allegro/result_2.log -t b test_allegro/result_2.log 2 2> test_allegro/error_2.log
python petrims.py -c -p test_allegro/result_3.log -t b test_allegro/result_3.log 2 2> test_allegro/error_3.log
python petrims.py -c -p test_allegro/result_4.log -t b test_allegro/result_4.log 2 2> test_allegro/error_4.log
python petrims.py -c -p test_allegro/result_5.log -t b test_allegro/result_5.log 2 2> test_allegro/error_5.log
python petrims.py -c -p test_allegro/result_6.log -t b test_allegro/result_6.log 2 2> test_allegro/error_6.log
python petrims.py -c -p test_allegro/result_7.log -t b test_allegro/result_7.log 2 2> test_allegro/error_7.log
python petrims.py -c -p test_allegro/result_8.log -t b test_allegro/result_8.log 2 2> test_allegro/error_8.log
python petrims.py -c -p test_allegro/result_9.log -t b test_allegro/result_9.log 2 2> test_allegro/error_9.log
python petrims.py -c -p test_allegro/result_10.log -t b test_allegro/result_10.log 2 2> test_allegro/error_10.log