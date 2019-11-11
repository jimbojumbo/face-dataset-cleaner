'''Copy original dataset to new destination according to clean image list
'''
import os
import csv
import time
import click

import queue, threading
import shutil


class ThreadedCopy(object):
    def __init__(self, source_path, dest_path, file_list_csv):
        self.file_queue = queue.Queue()

        file_list = []
        sub_folder_list = []
        with open(file_list_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    file_list.append([
                        os.path.join(source_path, row[1], row[2]),
                        os.path.join(dest_path, row[1], row[2])
                    ])
                    sub_folder_list.append(os.path.join(dest_path, row[1]))
                line_count += 1

        for sub_folder in list(set(sub_folder_list)):
            os.makedirs(sub_folder, exist_ok=True)

        print("total number of files to copy is {}.".format(len(file_list)))

        start_time = time.time()
        self.thread_worker(file_list)
        print('Took {:.3f} seconds to copy'.format(time.time() - start_time))

    def worker(self):
        while True:
            file_name = self.file_queue.get()
            shutil.copy(file_name[0], file_name[1])
            self.file_queue.task_done()

    def thread_worker(self, file_name_list):
        for i in range(16):
            t = threading.Thread(target=self.worker)
            t.daemon = True
            t.start()
        for file_name in file_name_list:
            self.file_queue.put(file_name)
        self.file_queue.join()


@click.command()
@click.option('--image-source-dir',
              type=click.Path(exists=True),
              help='Directory to the original face image folder',
              required=True)
@click.option(
    '--image-destination-dir',
    type=click.Path(exists=False),
    help='Directory to the destination where cleaned face image will be saved')
@click.option('--image-list-csv',
              type=click.Path(exists=True),
              help='CSV file that the clean image list is stored')
def cli(image_source_dir, image_destination_dir, image_list_csv):
    source_path = image_source_dir
    dest_path = image_destination_dir
    file_list_csv = image_list_csv

    ThreadedCopy(source_path, dest_path, file_list_csv)


if __name__ == '__main__':
    cli()