import os
import argparse
import numpy as np

from sqldb.utils import create_connection, create_table, insert
from imglib.utils import VGG, PCA, KDTree


def prepare_docs(conn, table, doc_path, n=None, check_before_insert=False):
    listdir = os.listdir(doc_path)
    if n is None:
        n = len(listdir)
    listdir = listdir[:n]

    model = VGG(top=1)
    vecs = []
    hashes = []
    files = []
    for fname in listdir:
        doc_file = os.path.join(doc_path, fname)
        # skip dirs
        if not os.path.isfile(doc_file):
            continue
        vecs.append(model.predict(doc_file))
        hashes.append(model.get_hash(doc_vec))
        files.append(doc_file)
    return vecs, hashes, files


def get_kwargs(kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', dest='db', action='store', help='/path/to/db_file')
    parser.add_argument('--image-path', dest='image_path', action='store', help='/path/to/image_dir')
    parser.add_argument('--drop-prev-db', dest='drop_prev_db', action='store_true')
    parser.add_argument('--output-dir', dest='output_dir', action='store', help='/path/to/output_dir')
    parser.add_argument('--table', dest='table', action='store', help='table name', default='myapp_images')
    for key, value in parser.parse_args().__dict__.items():
        kwargs[key] = value


def main(*kargs, **kwargs):
    get_kwargs(kwargs)
    db_file = kwargs['db']
    img_path = kwargs['image_path']
    drop_prev_db = kwargs['drop_prev_db']
    output_dir = kwargs['output_dir']
    table = kwargs['table']

    conn = create_connection(db_file)

    if drop_prev_db:
        sql_drop_table = 'DROP TABLE IF EXISTS {}'.format(table)
        create_table(conn, sql_drop_table)

    sql_create_table = """ CREATE TABLE IF NOT EXISTS {} (
                                        id integer PRIMARY KEY AUTOINCREMENT,
                                        file text NOT NULL,
                                        hash text NOT NULL,
                                        vec array NOT NULL,
                                        vec_pca array NOT NULL
                                    ); """.format(table)

    create_table(conn, sql_create_table)
    vecs, hashes, files = prepare_docs(img_path)

    pca = PCA(n_components=300)
    vecs_pca = pca.fit_transform(vecs)
    pca.save(os.path.join(output_dir, 'pca.model'))

    kdt = KDTree(vecs_pca, leaf_size=100, metric='minkowski')
    kdt.save(os.path.join(output_dir, 'kdtree.model'))

    create_table(conn, sql_create_table)
    insert(conn, table, ['file', 'hash', 'vec', 'vec_pca'], np.array([files, hashes, vecs, vecs_pca]).T)


if __name__=='__main__':
    main()
