'''Build adjacency information
Iterate through all the face image pairs within each identities to build adjacency matrix of a graph, 
the weight of each edge is the similarity score between the image pair.

Takes a face embedding json file as input, the layout of the json file is 
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

Outputs the image pair information saved as csv file for each identity in the image folder
'''
import os
import csv
import json
import click

import numpy as np

from tqdm import tqdm


@click.command()
@click.option(
    '--image-embedding-json',
    type=click.Path(exists=True),
    help='Path to the embeddings of all the images that will be cleaned',
    required=True)
@click.option(
    '--save-csv-dir',
    type=click.Path(exists=False),
    help='Directory where the image csv files will be saved',
    required=True)
def cli(image_embedding_json, save_csv_dir):
    os.makedirs(save_csv_dir, exist_ok=True)
    face_embeddings = json.load(open(image_embedding_json, 'r'))

    for id_name in tqdm(face_embeddings.keys()):
        image_csv_path = os.path.join(save_csv_dir, id_name + '.csv')
        image_list = list(face_embeddings[id_name].keys())

        with open(image_csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'file1', 'file2', 'score'])
            # Get embeddings for all the face images under one id_name
            _embeddings = []
            for image_name in image_list:
                _embeddings.append(face_embeddings[id_name][image_name])

            _embeddings = np.array(_embeddings)
            similarities = np.matmul(_embeddings, _embeddings.T)

            assert similarities.shape[0] == len(image_list)

            # Write the node (image) and edge (similarity) information to CSV file
            for i in range(len(image_list) - 1):
                for j in range(i + 1, len(image_list)):
                    img1 = image_list[i]
                    img2 = image_list[j]

                    # Calculate cosine similarity score between all image pairs
                    ang = similarities[i, j].astype('float32')
                    writer.writerow([id_name, img1, img2, ang])


if __name__ == '__main__':
    cli()
