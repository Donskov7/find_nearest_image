import time
from imglib.utils import find_nearest_doc
from sqldb.utils import get_exist_docs

def search_nearest(db_file, table, fname, model, pca, kdt):
    curr_doc = {}
    curr_doc['file'] = fname
    curr_doc['vec'] = model.predict(curr_doc['file'])
    curr_doc['hash'] = model.get_hash(curr_doc['vec'])
    if curr_doc['vec'] is not None:
        vec_pca = pca.transform([curr_doc['vec']])
        idxs = kdt.query(vec_pca, k=5)[1][0]
        idxs = [str(idx+1) for idx in idxs]
        existed_docs = get_exist_docs(db_file, table, where='id in ({})'.format(','.join(idxs)))
        nearest_doc_info = find_nearest_doc(curr_doc, existed_docs)
        return nearest_doc_info
    return None
