import time
from imglib.utils import find_nearest_doc
from sqldb.utils import get_exist_docs

def search_nearest(db_file, table, fname, model, pca, kdt):
    curr_doc = {}
    curr_doc['file'] = fname
    start2 = time.time()
    curr_doc['vec'] = model.predict(curr_doc['file'])
    print('vec', time.time() - start2)
    curr_doc['hash'] = model.get_hash(curr_doc['vec'])
    if curr_doc['vec'] is not None:
        start3 = time.time()
        vec_pca = pca.transform([curr_doc['vec']])
        print('pca', time.time() - start3)
        start4 = time.time()
        idxs = kdt.query(vec_pca, k=5)[1][0]
        print('kdt', time.time() - start4)
        start5 = time.time()
        idxs = [str(idx+1) for idx in idxs]
        print('idxs', time.time() - start5)
        start6 = time.time()
        existed_docs = get_exist_docs(db_file, table, where='id in ({})'.format(','.join(idxs)))
        print('get_docs', time.time() - start6)
        start7 = time.time()
        nearest_doc_info = find_nearest_doc(curr_doc, existed_docs)
        print('find_nearest', time.time() - start7)
        print(nearest_doc_info['sim'])
        del nearest_doc_info['sim']
        return nearest_doc_info
    return None
