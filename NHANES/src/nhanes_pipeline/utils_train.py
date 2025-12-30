from .core import lib

def make_dataset(
    real_data_dir: str,
    T: lib.Transformations,
):
    X_cat = {}
    X_num = {}
    y = {}
    for split in ['train', 'val', 'test']:
        X_num_t, X_cat_t, y_t = lib.read_pure_data(real_data_dir, split)
        X_num[split] = X_num_t
        X_cat[split] = X_cat_t
        y[split] = y_t
    D = lib.Dataset(
        X_num,
        X_cat,
        y,
    )
    return lib.transform_dataset(D, T)

