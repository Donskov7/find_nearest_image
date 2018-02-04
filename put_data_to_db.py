import os
import argpasre

from sqldb.utils import create_connection, create_table, adapt_array, convert_array, insert
from imglib.utils import VGG


def insert_docs(conn, table, doc_path, n=None, check_before_insert=False):
    listdir = os.listdir(doc_path)
    if n is None:
        n = len(listdir)
    listdir = listdir[:n]

    hash_existed = set()
    if check_before_insert:
        for row in conn.execute('SELECT hash FROM {}'.format(table)):
            hash_existed.add(row[0])

    model = VGG(top=1)
    to_insert = []
	i = 0
    for fname in listdir:
        doc_file = os.path.join(doc_path, fname)
		# skip dirs
        if not os.path.isfile(doc_file):
            continue
        doc_vec = model.predict(doc_file)
        doc_alias = model.predict(doc_file, True)[0][1]
        doc_hash = model.get_hash(doc_vec)
		# skip already existed images
        if doc_hash in hash_existed:
            continue
        to_insert.append([doc_file, str(doc_hash), doc_vec, doc_alias])
		i += 1
        if (i % 100 == 0 or i == n) and to_insert:
            insert(conn, table, ['file', 'hash', 'vec', 'alias'], to_insert)
            to_insert = []
			print('{} images have added to db'.format(i))


def get_kwargs(kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', dest='db', action='store', help='/path/to/db_file')
    parser.add_argument('--image-path', dest='image_path', action='store', help='/path/to/image_dir')
	parser.add_argument('--drop-prev-db', dest='drop_prev_db', action='store_true')
    for key, value in parser.parse_args().__dict__.items():
        kwargs[key] = value


def main(*kargs, **kwargs):
    get_kwargs(kwargs)
    db_file = kwargs['db']
    img_path = kwargs['image_path']
	drop_prev_db = kwargs['drop_prev_db']

    conn = create_connection(db_file)

	if drop_prev_db:
		sql_drop_table = 'DROP TABLE IF EXISTS myapp_images'
		create_table(conn, sql_drop_table)

    sql_create_table = """ CREATE TABLE IF NOT EXISTS myapp_images (
                                        id integer PRIMARY KEY AUTOINCREMENT,
                                        file text NOT NULL,
                                        hash text NOT NULL,
                                        vec array NOT NULL,
                                        alias text NOT NULL
                                    ); """

    create_table(conn, sql_create_table)
    insert_docs(conn, 'myapp_images', img_path, check_before_insert=True)


if __name__=='__main__':
    main()
