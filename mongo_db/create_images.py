
import numpy as np
import bsonnumpy
import utils
from multiprocessing import Pool
from config import SYMBOLS, n_workers, step


def filter_draw(symbol):
    collection = utils.get_db_collection()
    dtype = np.dtype([('date', np.int64),('Open',np.double),('High',np.double),('Low',np.double),('Close',np.double)])
    filter = {"symbol":symbol}
    ndarray = bsonnumpy.sequence_to_ndarray(collection.find_raw_batches(filter), dtype, collection.count(filter))
    utils.create_history_file_images(ndarray = ndarray,symbol = symbol,step = step)

if __name__ == "__main__":
    with Pool(processes = n_workers) as pool:
        pool.map(filter_draw, SYMBOLS)
    


