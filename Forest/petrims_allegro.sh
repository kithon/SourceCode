#/bin/sh

# allegro
# 1～10 : not overlap
python petrims.py -c test_allegro/result_1.log 2 2> test_allegro/error_1.log
python petrims.py -c test_allegro/result_2.log 2 2> test_allegro/error_2.log
python petrims.py -c test_allegro/result_3.log 2 2> test_allegro/error_3.log
python petrims.py -c test_allegro/result_4.log 2 2> test_allegro/error_4.log
python petrims.py -c test_allegro/result_5.log 2 2> test_allegro/error_5.log
python petrims.py -c test_allegro/result_6.log 2 2> test_allegro/error_6.log
python petrims.py -c test_allegro/result_7.log 2 2> test_allegro/error_7.log
python petrims.py -c test_allegro/result_8.log 2 2> test_allegro/error_8.log
python petrims.py -c test_allegro/result_9.log 2 2> test_allegro/error_9.log
python petrims.py -c test_allegro/result_10.log 2 2> test_allegro/error_10.log
# 11～20 : overlap
python petrims.py test_allegro/result_11.log 2> test_allegro/error_11.log
python petrims.py test_allegro/result_12.log 2> test_allegro/error_12.log
python petrims.py test_allegro/result_13.log 2> test_allegro/error_13.log
python petrims.py test_allegro/result_14.log 2> test_allegro/error_14.log
python petrims.py test_allegro/result_15.log 2> test_allegro/error_15.log
python petrims.py test_allegro/result_16.log 2> test_allegro/error_16.log
python petrims.py test_allegro/result_17.log 2> test_allegro/error_17.log
python petrims.py test_allegro/result_18.log 2> test_allegro/error_18.log
python petrims.py test_allegro/result_19.log 2> test_allegro/error_19.log
python petrims.py test_allegro/result_20.log 2> test_allegro/error_20.log
