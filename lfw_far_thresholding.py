'''
Determine the threshold that achieve a pre-defined FAR for LFW validation result

Take pre-prepared LFW embedding json file as input, in the structure of 
{
    id_1 : 
    {
        {image _1 : embedding}
        {image _2 : embedding}
        ...
    }
    id_2 : 
    {
        {image _1 : embedding}
        {image _2 : embedding}
        ...
    }
    ...
}.

Also takes LFW pair txt file as input to find all the validation pairs to determine the threshold

Used FAR = 0.001 as mentioned in the paper to find the threshold.
'''
import json
import click

import numpy as np


def _get_negative_pair_info(lfw_dict, pairs):

    source_list = []
    target_list = []
    score_list = []

    for pair in pairs:
        if len(pair) == 4:
            id_name_1 = pair[0]
            image_name_1 = pair[0] + '_' + '%04d' % int(pair[1]) + '.jpg'
            descriptor_1 = np.array(
                lfw_dict[id_name_1][image_name_1]['descriptor'])

            id_name_2 = pair[2]
            image_name_2 = pair[2] + '_' + '%04d' % int(pair[3]) + '.jpg'
            descriptor_2 = np.array(
                lfw_dict[id_name_2][image_name_2]['descriptor'])

            similarity = abs(np.matmul(descriptor_1, descriptor_2.T))

            source_list.append(id_name_1)
            target_list.append(id_name_2)
            score_list.append(similarity)

    return source_list, target_list, score_list


def _read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


@click.command()
@click.option(
    '--lfw-embedding-json',
    type=click.Path(exists=True),
    help='Path to the json file that stores face embeddings for LFW dataset',
    required=True)
@click.option(
    '--lfw-pair-txt',
    type=click.Path(exists=False),
    help='TXT file used for LFW evaluation purpose')
@click.option(
    '--target-far',
    type=float,
    help='FAR used to determine the threshold',
    default=0.001)
def cli(lfw_embedding_json, lfw_pair_txt, target_far):
    lfw_descriptor = json.load(open(lfw_embedding_json, 'r'))
    lfw_pairs = _read_pairs(lfw_pair_txt)

    source_list, target_list, score_list = _get_negative_pair_info(
        lfw_descriptor, lfw_pairs)
    scores = np.array(score_list)

    thresholds = np.linspace(0, 1, 10000)
    closest = 10
    found_threshold = 0

    for th in thresholds:
        tnr = np.sum(np.less_equal(scores, th)) / len(scores)
        far = 1 - tnr

        if abs(far - target_far) <= closest:
            closest = abs(far - target_far)
            found_threshold = th

    print('The threshold is {} for FAR {}'.format(found_threshold, target_far))


if __name__ == '__main__':
    cli()
