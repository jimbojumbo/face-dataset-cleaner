'''Clean face dataset based on adjacency and community information
An implementation of the paper "A Community Detection Approach to Cleaning Extremely Large Face Database"

Run dataset_adjacency_build.py first, and use folder that contains all the csv files as input

Outputs two csv files contains the cleaned image names and noisy image names after the process

The original paper selects p = 0.1 as small community size
'''
import os
import click

import pandas as pd
import numpy as np

from tqdm import tqdm
from pathlib import Path

import igraph as ig
import louvain


def list_csv(root_dir):
    return [
        Path(root_dir).joinpath(f) for f in Path(root_dir).iterdir()
        if f.suffix in ['.csv', '.CSV']
    ]


def get_node_edge(dataframe):
    # Acquire node edge file that can be visualized in Gephi
    image_names = dataframe.file1.unique().tolist()
    target_names = dataframe.file2.unique().tolist()
    image_names.append(target_names[-1])

    image_label_mapping = dict(
        [[image_names[i], i] for i in range(len(image_names))])

    image_numerical_label = list(
        map(lambda x: image_label_mapping[x], image_names))
    node_df = pd.DataFrame(
        zip(image_numerical_label, image_names), columns=['Id', 'Label'])

    source = list(map(lambda x: image_label_mapping[x], list(dataframe.file1)))
    target = list(map(lambda x: image_label_mapping[x], list(dataframe.file2)))
    edge_type = ['Undirected'] * len(source)
    weight = list(abs(dataframe.score))

    edge_df = pd.DataFrame(
        zip(source, target, edge_type, weight),
        columns=['Source', 'Target', 'Type', 'Weight'])

    return node_df, edge_df


def community_partition(graph, minimum_community):
    # Perform the graph partition to get the community
    partition = louvain.find_partition(graph,
                                       louvain.ModularityVertexPartition)

    major, minor = [], []
    for i in range(len(partition)):
        _g = partition.subgraphs()[i]
        if len(_g.vs) > minimum_community:
            for v in _g.vs:
                major.append(v['name'])
        else:
            for v in _g.vs:
                minor.append(v['name'])

    return major, minor


@click.command()
@click.option(
    '--graph-info-dir',
    type=click.Path(exists=True),
    help=
    'Directory to the CSV files that contains node and edge information of dataset',
    required=True)
@click.option(
    '--cleaned-lists-dir',
    type=click.Path(exists=False),
    help='Directory to the clean image and noise image result will be saved')
@click.option(
    '--threshold',
    type=float,
    help=
    'Threshold used to break the edge between nodes, obtained through lfw_far_thresholding.py',
    required=True)
@click.option(
    '--p',
    type=float,
    help='Minimum percentage of vertexes to consider a valid community',
    default=0.1)
def cli(graph_info_dir, cleaned_lists_dir, threshold, p):
    graph_csv_dir = Path(graph_info_dir)
    cleaned_result_save_dir = Path(cleaned_lists_dir)
    os.makedirs(cleaned_result_save_dir, exist_ok=True)
    # List all the csv file that contains the node, edge information for each person
    face_score_csv = list_csv(graph_csv_dir)
    # Clean the dataset with community information
    clean_list = []
    noise_list = []
    for file_name in tqdm(face_score_csv):
        class_name = file_name.stem

        df = pd.read_csv(
            file_name,
            header=0,
            names=['ID', 'file1', 'file2', 'score'],
            dtype={
                'ID': str,
                'file1': str,
                'file2': str,
                'score': np.float32,
            })

        image_names = df.file1.unique().tolist()
        target_names = df.file2.unique().tolist()
        image_names.append(target_names[-1])
        # Minimum valid community size
        minimum_community = int(np.floor(len(image_names) * p))

        image_label_mapping = dict(
            [[image_names[i], i] for i in range(len(image_names))])
        # Give all the images in one folder a numerical label
        source = list(map(lambda x: image_label_mapping[x], list(df.file1)))
        target = list(map(lambda x: image_label_mapping[x], list(df.file2)))
        weight = list(abs(df.score))

        #node_df, edge_df = get_node_edge(df)
        #node_df.to_csv(data_dir.joinpath('node.csv'), index=False)
        #edge_df.to_csv(data_dir.joinpath('edge.csv'), index=False)

        del df

        # Initialize the Graph with image name attached to each node
        G = ig.Graph()
        G.add_vertices(len(image_names))
        G.vs['name'] = image_names

        edges = []
        count = 0
        for s, t in zip(source, target):
            # Add edge that above pre-defined threshold
            if weight[count] > threshold:
                edges.append((s, t))
            count += 1
        G.add_edges(edges)

        good_images, bad_images = community_partition(G, minimum_community)
        clean_list.extend([[class_name, goody] for goody in good_images])
        noise_list.extend([[class_name, baddy] for baddy in bad_images])

    clean_df = pd.DataFrame(clean_list, columns=['ID', 'file'])
    noise_df = pd.DataFrame(noise_list, columns=['ID', 'file'])
    clean_df.to_csv(
        cleaned_result_save_dir.joinpath('clean_list.csv'),
        header=['ID', 'file'])
    noise_df.to_csv(
        cleaned_result_save_dir.joinpath('noise_list.csv'),
        header=['ID', 'file'])


if __name__ == '__main__':
    cli()