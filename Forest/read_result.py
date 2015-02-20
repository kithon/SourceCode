# -*- coding: utf-8 -*-
import os
import commands
import linecache
import numpy as np
from ast import literal_eval

def etrims_score(file_name):
    awk_tree = "awk '/train/ && /Tree/ {print $3}' %s" % file_name
    #awk_forest = "awk '/train/ && /Forest/ {print $3}' %s" % file_name
    awk_score = "awk '/score/ {print $3, $6}' %s" % file_name
    awk_node = "awk '/Information/ && /node/ {print $8}' %s" % file_name
    
    tree_list = commands.getoutput(awk_tree).split('\n')

    index = 0
    score_list = []
    temp_list = commands.getoutput(awk_score).split('\n')
    for l in temp_list:
        char = l.split(' ')
        if char[0] == 'depth:0':
            index = len(score_list)
            score_list.append([])
        score_list[index].append(float(char[1]))

    depth_list = []
    for l in score_list:
        depth_list.append(len(l))
        
    node_list  = []
    temp_list = commands.getoutput(awk_node).split('\n')
    for l in temp_list:
        node_list.append(int(l))
    
    return [tree_list, score_list, depth_list, node_list]

def score_length(score_list, depth):
    count = 0
    for score in score_list:
        if len(score) > depth:
            count += 1
    return count

def mean_score(score_list, depth):
    temp_list = []
    for score in score_list:
        if len(score) > depth:
            temp_list.append(score[depth])
        else:
            temp_list.append(score[-1])
    return np.average(temp_list)

def var_score(score_list, depth):
    temp_list = []
    for score in score_list:
        if len(score) > depth:
            temp_list.append(score[depth])
        else:
            temp_list.append(score[-1])
    return np.var(temp_list)

def mean_depth(depth_list):
    return np.average(depth_list)

def var_depth(depth_list):
    return np.var(depth_list)

def mean_node(node_list):
    return np.average(node_list)

def var_node(node_list):
    return np.var(node_list)    
    
if __name__ == '__main__':
    # get data from file
    dir_list = ['test_dir/test_andante/',
                'test_dir/test_allegro/',
                'test_dir/test_dacapo/']
    
    detail_list = []
    for d in dir_list:
        file_list = commands.getoutput('ls %s' % d).split('\n')
        for f in file_list:
            path = '%s%s' % (d,f)
            print path
            detail_list.append(etrims_score(path))
            
    # reform data
    tree_set = set()
    detail_dic = {}
    for detail in detail_list:
        tree_list, score_list, depth_list, node_list = detail
        for i,tree in enumerate(tree_list):
            if not tree in tree_set:
                tree_set.add(tree)
                detail_dic[tree, 'score'] = []
                detail_dic[tree, 'depth'] = []
                detail_dic[tree, 'node' ] = []
            detail_dic[tree, 'score'].append(score_list[i])
            detail_dic[tree, 'depth'].append(depth_list[i])
            detail_dic[tree, 'node' ].append(node_list[i])

            
    for tree in tree_set:
        print tree
        temp = detail_dic[tree, 'score']
        depth = 0
        while True:
            length = score_length(temp, depth)
            if length == 0:
                break
            mean = mean_score(temp, depth) 
            var = var_score(temp, depth)
            print "score depth:", depth, "length:", length, "mean:", mean, "var:", var
            depth += 1

        mean, var = mean_score(temp, -1), var_score(temp, -1)
        print "score terminal mean:", mean, "var:", var

        temp = detail_dic[tree, 'depth']
        mean, var = mean_depth(temp), var_depth(temp)
        print "depth mean:", mean, "var:", var

        temp = detail_dic[tree, 'node']
        mean, var = mean_node(temp), var_node(temp)
        print "node mean:", mean, "var:", var

    
