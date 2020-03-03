import glob
import os, shutil
import math
from itertools import permutations
from a1 import *


shapes = {
    'scc',
    'triangle',
    'circle',
    'dot',
    'square',
    'rectangle'
}

hlocs = {
    'left',
    'center',
    'right'
}

vlocs = {
    'top',
    'middle',
    'bottom',
}

rels = {
    'left_of',
    'right_of',
    'above',
    'below',
    'overlap',
    'inside'
}

penalty = {
    'position': {
        'no_change' : 0,
        'move' : 10,
    },
    'relation' : {
        'no_change' : 0,
        'change' : 5,
    },
    'object' : {
        'no_change' : 0,
        'change' : {
            'size' : 15,
            'shape' : 40,
            'both' : 55,
        },
        'add' : 20,
        'delete' : 20,
    },
    'corresp': {
        'add_move' : 40,
        'remove_move' : 40,
        'add_obj' : 55,
        'remove_obj' : 55,
        'rel' : 5,
    }
}

class interpret:

    def __init__(self, interp_name, parsed_dict):
        
        self.name = interp_name
        self.shapes = []
        self.shape_names = []
        self.rel = []
        
        for key, value in parsed_dict.items():
            self.shapes.append(shape(value))
            self.shape_names.append(value['name'])
            for rel in value['ext']:
                if type(rel) == type([]):
                    self.rel.append(rel)


class shape:

    def __init__(self, properties):
    
        self.name = properties['name']
        self.rel = []
        self.size = None

        for prop in properties['int']:
            if prop in shapes:
                self.shape = prop
            else:
                self.size = prop

        for prop in properties['ext']:
            if type(prop) == type([]):
                continue
            if prop in hlocs:
                self.hloc = prop
            elif prop in vlocs:
                self.vloc = prop


def transform(x, y):

    x_counter = 0
    y_counter = 0
    con_dict = {}
    
    while len(x.shapes) < len(y.shapes):
        x_counter += 1
        x.shapes.append(None)
        x.shape_names.append(None)

    while len(y.shapes) < len(x.shapes):
        y_counter += 1
        y.shapes.append(None)
        y.shape_names.append(None)

    perm = [list(zip(x.shape_names, p)) for p in permutations(y.shape_names)]

    for x_shape in x.shapes:
        for y_shape in y.shapes:
            if x_shape:
                x_name = x_shape.name
            else:
                x_name = None

            if y_shape:
                y_name = y_shape.name
            else:
                y_name = None

            con_dict[(x_name, y_name)] = {}
            transform_to(x_shape, y_shape, con_dict[(x_name, y_name)])

    while x_counter > 0:
        x_counter -= 1
        del x.shapes[-1]
        del x.shape_names[-1]

    while y_counter > 0:
        y_counter -= 1
        del y.shapes[-1]
        del y.shape_names[-1]
        

    transform_list = []

    for objects in perm:
        f_transf = {
            'shapes' : [],
            'changes' : [],
            'cost' : 0
        }

        for object in objects:
            f_transf['shapes'].append(object)
            f_transf['changes'] += con_dict[object]['changes']
            f_transf['cost'] += con_dict[object]['cost']
        
        transform_rel(x, y, f_transf)
        transform_list.append(f_transf.copy())
   
    return transform_list

        
def transform_rel(x, y, f_transf):

    def eq_rel(comp1, comp2, sent1, sent2):

        if comp1 == comp2 and sent1 == sent2:
            return True
            
        lsent1 = sent1.split(' ')
        lsent2 = sent2.split(' ')

        if (comp1 != comp2 or comp1 == comp2 == 'overlap') and (lsent1[0] == lsent2[1] and lsent1[1] == lsent2[0]):
            return True
        
        return False

    subs = {}
    for shape in f_transf['shapes']:
        subs[shape[0]] = shape[1]

    cost = 0
    description = []
    
    x_rel = x.rel
    y_rel = y.rel

    min_i = min(len(x_rel), len(y_rel))
    for i in range(0, min_i):
        sent1 = x_rel[i][1]
        sent2 = y_rel[i][1]

        for key in subs.keys():
            sent1 = sent1.replace(key, subs[key])

        if not eq_rel(x_rel[i][0], y_rel[i][0], sent1, sent2):
            cost += penalty['relation']['change']
            description.append([
                'move',
                x_rel[i],
                y_rel[i]
            ])

    f_transf['changes'] += description.copy()
    f_transf['cost'] += cost
    

def transform_to(x, y, con_dict):

    mismatch = 2

    cost = 0
    description = []

    con_dict['cost'] = 0
    con_dict['changes'] = []
    
    # if None -> Shape, shape is added
    if not x:
        cost += penalty['object']['add']
        description.append(['add', y.name])
        con_dict['cost'] = cost
        con_dict['changes'] = description.copy()
        return

    # if Shape -> None, shape is deleted    
    if not y:
        cost += penalty['object']['delete']
        description.append(['delete', x.name])
        con_dict['cost'] = cost
        con_dict['changes'] = description.copy()
        return
        
    x_name = x.name
    y_name = y.name
    
    if x_name[1] != y_name[1]:
        cost += mismatch

    # Compare shape
    if x.shape != y.shape:
        cost += penalty['object']['change']['shape']
        description.append([
            'change',
            x.shape,
            x_name,
            y.shape,
            y_name
        ])

    # Compare size if they exist and shapes match
    if x.size and y.size:
        if x.size == y.size:
            cost += penalty['object']['no_change']
        
        else:
            cost += penalty['object']['change']['size']
            description.append([
                'change',
                x.size,
                x_name,
                y.size,
                y_name
            ])
    
    # Compare hloc
    if x.hloc != y.hloc:
        cost += penalty['position']['move']
        description.append([
            'move',
            x.hloc,
            x_name,
            y.hloc,
            y_name
        ])

    else:
        cost += penalty['position']['no_change']
    
    # Compare vloc
    if x.vloc != y.vloc:
        cost += penalty['position']['move']
        description.append([
            'move',
            x.vloc,
            x_name,
            y.vloc,
            y_name
        ])

    else:
        cost += penalty['position']['no_change']
     
    con_dict['cost'] = cost
    con_dict['changes'] = description.copy()
    

def write_interprets(interpret_dict, used_interprets, outputFile):

    used_set = set(used_interprets)

    for object, attributes in interpret_dict.items():
        
        if object in used_set:
            outputFile.write(object + '\n')
            for property in attributes:
                outputFile.write(property + '\n')



def readInput(inputFile, objectName, interpret_dict):

    if objectName not in interpret_dict:
        interpret_dict[objectName] = []
    
    with open(inputFile) as inputText:
        
        for i, row_in in enumerate(inputText):
            if row_in.find('=') != -1 or row_in.find(':') != -1:
                continue

            row = row_in.replace('\n','')

            interpret_dict[objectName].append(row)


def parse_interpret(interpret_dict, parsed_data):

    entries = ['A', 'B', 'C', 'K1', 'K2', 'K3', 'K4', 'K5']
    
    for entry in entries:
        parsed_data[entry] = []

    def init_dict(dict, rest, tag, counter):

        dict[rest] = {}
        to_construct[rest]['name'] = '{}{}'.format(tag, counter)
        
        if 'int' not in dict[rest]:
            dict[rest]['int'] = []
            
        if 'ext' not in dict[rest]:
            dict[rest]['ext'] = []
        
        return counter + 1
    
    object_word = {
        'vloc',
        'hloc'
    }

    double_object = {
        'right_of',
        'left_of',
        'inside',
        'above',
        'below',
        'overlap'
    }

    for object, attributes in interpret_dict.items():
        to_construct = {}
        tag = object[0].lower()
        counter = 1

        for property in attributes:
            
            split_up = property.split('(')
            check = split_up[0]
            rest = split_up[1].replace(')','')

            if check in object_word:
                rest = rest.split(',')
                if rest[0] not in to_construct:
                    counter = init_dict(to_construct, rest[0], tag, counter)
                
                to_construct[rest[0]]['ext'].append(rest[1])
            
            elif check in double_object:
                rest = rest.split(',')

                if rest[0] not in to_construct:
                    counter = init_dict(to_construct, rest[0], tag, counter)

                if rest[1] not in to_construct:
                    counter = init_dict(to_construct, rest[1], tag, counter)
                    
                to_construct[rest[0]]['ext'].append([
                    check,
                    '{} {}'.format(
                        to_construct[rest[0]]['name'],
                        to_construct[rest[1]]['name']
                    )
                ])
            
            else:
                if rest not in to_construct:
                    counter = init_dict(to_construct, rest, tag, counter)
                    
                to_construct[rest]['int'].append(check)
                
        if object[0] == 'K':
            parsed_data[object[0:2]].append(interpret(object, to_construct))
        
        else:
            parsed_data[object[0]].append(interpret(object, to_construct))
        

def compute_transformation(parsed_data, transform_dict):

    for x in parsed_data['A']:
        for y in parsed_data['B']:
            make_entry(transform_dict['A -> B'], x, y)

    for x in parsed_data['C']:
        counter = 0
        while counter < 5:
            counter += 1
            for y in parsed_data['K{}'.format(counter)]:
                make_entry(transform_dict['C -> K{}'.format(counter)], x, y)
    

def print_transforms(transform_dict):

    for k, v in transform_dict.items():
        print(k)
        for ki, vi in v.items():
            print('\t{}'.format(ki))
            for vii in vi:
                print('\t\t{}\n'.format(vii['shapes']))
                print('\t\t{}\n'.format(vii['cost']))
                for xx in vii['changes']:
                    print('\t\t{}'.format(xx))
                print('')
            print('')

            
def make_entry_2nd(final_transf_dict, key_ab, key_ck, x, y):

    if (key_ab, key_ck) not in final_transf_dict:
        final_transf_dict[
        (
            key_ab,
            key_ck
        )
    ] = []

    final_transf_dict[
        (
            key_ab,
            key_ck
        )
    ].append(second_transform(x, y))


def compute_2nd_transform(transform_dict, final_transf_dict):

    for key_ab, v_ab in transform_dict['A -> B'].items():
        for transf_ab in v_ab:
            counter = 0
            while counter < 5:
                counter += 1
                for key_ck, v_ck in transform_dict['C -> K{}'.format(counter)].items():
                    for transf_ck in v_ck:
                        make_entry_2nd(
                            final_transf_dict[counter],
                            key_ab,
                            key_ck,
                            transf_ab,
                            transf_ck
                        )

                        
def transform_to_2nd(x, y, f_transf, shape_list):

    changes1 = set()
    changes2 = set()

    for y_chng in y['changes']:
        changes2.add(str(y_chng))
    
    subs = {}
    for shape in shape_list:
        if not all(shape[0]) and all(shape[1]):
            f_transf['changes'].append('add {}'.format(shape[1]))
            f_transf['cost'] += penalty['corresp']['add_obj']

        if not all(shape[1]) and all(shape[0]):
            f_transf['changes'].append('remove {}'.format(shape[1]))
            f_transf['cost'] += penalty['corresp']['remove_obj']

        if shape[0][0] is not None and shape[1][0] is not None:
            subs[shape[0][0]] = shape[1][0]
            f_transf['shapes'].append((shape[0][0], shape[1][0]))

        if shape[0][1] is not None and shape[1][1] is not None:
            subs[shape[0][1]] = shape[1][1]
            f_transf['shapes'].append((shape[0][1], shape[1][1]))


    for x_chng in x['changes']:
        sent = str(x_chng)
        for key in subs.keys():
            sent = sent.replace(key, subs[key])
        
        changes1.add(sent)
        
    for diff in changes1 - changes2:
        sent = diff
        for key in subs.keys():
            sent = sent.replace(subs[key], key)
        

        f_transf['changes'].append('remove ({})'.format(sent))
        if diff[2:6] != 'move':
            
            f_transf['cost'] += penalty['corresp']['remove_obj']
        else:
            if diff[9] == '[':
                f_transf['cost'] += penalty['corresp']['rel']
            else:
                f_transf['cost'] += penalty['corresp']['remove_move']
                 
    for diff in changes2 - changes1:
        f_transf['changes'].append('add ({})'.format(diff))
        if diff[2:6] != 'move':
            f_transf['cost'] += penalty['corresp']['add_obj']
        else:
            if diff[9] == '[':
                f_transf['cost'] += penalty['corresp']['rel']
            else:
                f_transf['cost'] += penalty['corresp']['add_move']


def second_transform(x , y):

    x_shapes = x['shapes']
    y_shapes = y['shapes']

    perm = [list(zip(x_shapes, p)) for p in permutations(y_shapes)]  

    transform_list = []

    for objects in perm:
        shape_list = []
        f_transf = {
            'shapes' : [],
            'changes' : [],
            'cost' : 0
        }

        for object in objects:
            shape_list.append(object)

        transform_to_2nd(x, y, f_transf, shape_list)
        transform_list.append(f_transf.copy())
   
    return transform_list


def make_entry(transform_dict, x, y):

    transform_dict[
        (
            x.name,
            y.name
        )
    ] = transform(x,y)


def compute_lowest_cost(transform_dict, final_transf_dict):

    min_cost = math.inf
    result_k = 0
    resulting_interps = []

    for key_ab, v_ab in transform_dict['A -> B'].items():
        for transf_ab in v_ab:
            counter = 0
            while counter < 5:
                counter += 1
                for key_ck, v_ck in transform_dict['C -> K{}'.format(counter)].items():
                    for transf_ck in v_ck:
                        min_meta_transform = math.inf
                        chosen_meta = {}
                        for s in second_transform(transf_ab, transf_ck):
                            if s['cost'] < min_meta_transform:
                                min_meta_transform = s['cost']
                                chosen_meta = s
                                
                        cost = transf_ab['cost'] + transf_ck['cost'] + chosen_meta['cost']

                        if cost < min_cost:
                            saved_ab = transf_ab
                            sabed_ck = transf_ck
                            saved_meta = chosen_meta
                            saved_count = counter
                            resulting_interps = [key_ab[0], key_ab[1], key_ck[0], key_ck[1]]
                            min_cost = cost
                            result_k = counter
                            
    print('K = {}'.format(result_k))
    # write_to_result(saved_ab, sabed_ck, saved_meta, cost, saved_count, outputFile)
    return resulting_interps


def write_to_result(dict1, dict2, dict3, cost, counter, outputFile):

    write_dict(dict1, outputFile)
    write_dict(dict2, outputFile)
    write_dict(dict3, outputFile)

def write_dict(dict, outputFile):

    outputFile.write('-----------------------------------------\n')
    outputFile.write(str(dict['shapes']) + '\n')
    for change in dict['changes']:
        outputFile.write(str(change) + '\n')
        
    
def remove_files(outputFolder):

    folder = outputFolder
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def main():

    inputFolder = sys.argv[1]
    outputFolder = 'new'


    inputFiles = glob.glob(inputFolder + '/*.txt')
    for filePath in inputFiles:
        filePath = filePath.replace('\\','/')
        a1(filePath, outputFolder)

    inputFiles = glob.glob(outputFolder + '/*.txt')
    objectName = ''
    interpret_dict = {}
    parsed_data = {}
    transform_dict = {
        'A -> B': {},
        'C -> K1' : {},
        'C -> K2' : {},
        'C -> K3' : {},
        'C -> K4' : {},
        'C -> K5' : {}
    }
    final_transf_dict = {
        1 : {},
        2 : {},
        3 : {},
        4 : {},
        5 : {}
    }

    for filePath in inputFiles:
    
        fileName = os.path.basename(filePath)
       
        fileName = fileName.split('.')[0]

        readInput(filePath, fileName, interpret_dict)

        parse_interpret(interpret_dict, parsed_data)

    compute_transformation(parsed_data, transform_dict)

    compute_2nd_transform(transform_dict, final_transf_dict)

    # outputFile = open('result.txt','w')
    
    used_interprets = compute_lowest_cost(transform_dict, final_transf_dict)
    
    # write_interprets(interpret_dict, used_interprets, outputFile)
    
    remove_files(outputFolder)

if __name__ == '__main__':
    main()
