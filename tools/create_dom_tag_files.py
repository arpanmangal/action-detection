"""
Script to create domain-wise TAG files to test model score on different domains
Usage:
$ python create_dom_tag_files.py <PATH_OF_TAXONOMY> <PATH_OF_DATASET_JSON> <PATH_OF_TAG_FILE>
"""

import os
import sys
import json
import pandas as pd

def read_block (tag_file):
    print ('Processing %s TAG file' % tag_file)
    f = open(tag_file, 'r')
    while (len(f.readline()) > 0):
        # Keep reading the block
        obj = {}
        obj['id'] = f.readline().strip().split('/')[-1]
        obj['frames'] = int(f.readline().strip())
        obj['task'] = int(f.readline().strip())
        obj['useless'] = f.readline()
        obj['correct'] = []
        corrects = int(f.readline().strip())
        for _c in range(corrects):
            annotation = f.readline().strip().split(' ')
            obj['correct'].append(annotation)
        obj['preds'] = []
        preds = int(f.readline().strip())
        for _p in range(preds):
            obj['preds'].append(f.readline().strip().split(' '))
        yield obj


def write_block (block, ofile, idx):
    with open(ofile, 'a') as f:
        f.write('# %d\n' % idx)
        f.write('%s\n' % block['id'])
        f.write('%d\n' % block['frames'])
        f.write('%d\n' % block['task'])
        f.write(block['useless'])
        corrects = block['correct']
        preds = block['preds']
        f.write('%d\n' % len(corrects))
        for c in corrects:
            f.write('%s\n' % ' '.join(c))
        f.write('%d\n' % len(preds))
        for p in preds:
            f.write('%s\n' % ' '.join(p))


if __name__ == '__main__':
    assert len(sys.argv) == 4
    taxonomy_file = sys.argv[1]
    dataset_file = sys.argv[2]
    tag_file = sys.argv[3]

    # Load the taxonomy and Add another column for domain id
    taxonomy = pd.read_csv(taxonomy_file)
    domains = list(set(taxonomy['Domains']))
    taxonomy['DomainID'] = taxonomy.apply(lambda row : domains.index(row['Domains']), axis = 1)
    task_to_domain = dict({row['Targets']:row['DomainID'] for _, row in taxonomy.iterrows()})
    new_taxonomy_file = taxonomy_file.split('.csv')[0] + '_new.csv'
    taxonomy.to_csv(new_taxonomy_file, index=False)

    # Load the dataset vidio to task map
    dataset = json.load(open(dataset_file, 'r'))['database']
    vid_to_task = {vid_id:info['class'] for vid_id, info in dataset.items()}
    for vid_id, task in vid_to_task.items():
        try:
            assert task in task_to_domain
        except:
            print ("Please correct the spelling of '%s' in taxonomy" % task)
    dataset_tasks = set({task for _, task in vid_to_task.items()})
    for task in task_to_domain:
        try:
            assert task in dataset_tasks
        except:
            print ("Task '%s' not in database" % task)

    # Make directory for dom_tags
    try:
        dom_tag_dir = os.path.join('/'.join(tag_file.split('/')[:-1]), 'dom_tags')
        os.mkdir(dom_tag_dir)
    except:
        pass

    # Write tag files
    tag_idxs = [0] * len(domains)
    dom_tag_files = []
    for i in range(len(domains)):
        dom_tag_file = os.path.join(dom_tag_dir, 'coin_tag_test%d_proposal_list.txt' % i)
        dom_tag_files.append(dom_tag_file)
        with open(dom_tag_file, 'w'): _overwritten = True

    for block in read_block(tag_file):
        vid_id = block['id']
        dom = task_to_domain[vid_to_task[vid_id]]
        tag_idxs[dom] += 1
        write_block(block, dom_tag_files[dom], tag_idxs[dom])
    
    print (domains)
    print (tag_idxs)
