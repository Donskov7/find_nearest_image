import sys
import os

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
    for i, fname in enumerate(listdir):
        doc_file = os.path.join(doc_path, fname)
        print(doc_file)
        if not os.path.isfile(doc_file):
            continue
        doc_vec = model.predict(doc_file)
        doc_alias = model.predict(doc_file, True)[0][1]
        print(doc_alias)
        doc_hash = model.get_hash(doc_vec)
        if doc_hash in hash_existed:
            print('Already exist')
            continue
        to_insert.append([doc_file, str(doc_hash), doc_vec, doc_alias])
        if (i % 100 == 0 or i == n - 1) and to_insert:
            insert(conn, table, ['file', 'hash', 'vec', 'alias'], to_insert)
            to_insert = []


def main():
    db_file = sys.argv[1]
    img_path = sys.argv[2]

    conn = create_connection(db_file)

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
