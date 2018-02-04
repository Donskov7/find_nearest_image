import sqlite3
import numpy as np
import io


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter("array", convert_array)
        conn = sqlite3.connect(db_file, detect_types=sqlite3.PARSE_DECLTYPES)
        return conn
    except Error as e:
        print(e)

    return None

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)

def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def insert(conn, table, columns, items):
    n = len(items[0])
    mask = ['?' for _ in range(n)]
    conn.executemany('INSERT INTO {}({}) VALUES ({})'.format(table, ','.join(columns), ','.join(mask)), items)
    conn.commit()

def get_column_names(db_file, table):
    conn = create_connection(db_file)
    cursor = conn.execute('select * from {}'.format(table))
    return list(map(lambda x: x[0], cursor.description))

def select_all(db_file, table, where):
    conn = create_connection(db_file)
    query = 'SELECT * FROM {}'.format(table)
    if where is not None:
        query = '{} WHERE {}'.format(query, where)
    for row in conn.execute(query):
        yield row

def get_exist_docs(db_file, table, where=None):
    result = []
    columns = get_column_names(db_file, table)
    for row in select_all(db_file, table, where):
        row = {col: val for col, val in zip(columns, row)}
        result.append(row)
    return result
