
import os
CUR_DIRECTORY = os.path.dirname(os.path.realpath(__file__)).replace('\\', '/')
SRC_DIRECTORY = os.path.dirname(CUR_DIRECTORY)
DATA_FILE = SRC_DIRECTORY + "/data/Viet74K.txt"
similar_pairs = [
    ('a', 'ă'), ('a', 'â'), ('ă', 'â'),  
    ('e', 'ê'),                         
    ('o', 'ô'), ('o', 'ơ'), ('ô', 'ơ'),  
    ('u', 'ư'),                          
    ('d', 'đ'),                         
    ('i', 'y'),                         
    ('s', 'x'),                          
    ('n', 'm'),                          
    ('t', 'th'),                        
    ('c', 'k'),                         
    ('g', 'gh'),                        
    ('ng', 'ngh'),                      
    ('h', 'k'),                          
    ('u', 'v'),                        
    ('o', '0'),                        
    ('l', '1'),                        
    ('i', 'l'),                         
    ('r', 'd'),                         
    ('g', 'q'),                          
    ('p', 'q'),                          
    ('a', 'e'),                          
    ('t', 'l'),                          
    ('b', 'v')                           
]
