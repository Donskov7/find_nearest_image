from imglib.utils import find_nearest_doc
from sqldb.utils import get_exist_docs

def search_nearest(db_file, table, fname, model):
    curr_doc = {}
    curr_doc['file'] = fname
    curr_doc['classes'] = model.predict(curr_doc['file'], full=True)
    curr_doc['classes'] = ['"{}"'.format(val[1]) for val in curr_doc['classes']]
    curr_doc['vec'] = model.predict(curr_doc['file'])
    curr_doc['hash'] = model.get_hash(curr_doc['vec'])
    if curr_doc['vec'] is not None:
        existed_docs = get_exist_docs(db_file, table, where='alias in ({})'.format(','.join(curr_doc['classes'])))
        nearest_doc_info = find_nearest_doc(curr_doc, existed_docs)
        print(nearest_doc_info['sim'])
        del nearest_doc_info['sim']
        return nearest_doc_info
    return None
