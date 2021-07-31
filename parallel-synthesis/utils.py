from cuml.preprocessing import StandardScaler
from cuml import PCA, I;
import matplotlib.pyplot as plt


def dim_reduction(X, y, strategy = 'PCA', n_components = 2, prefix = None, plot_viz = True, save_viz = False):
    valid_strats = ("PCA", "UMAP", "TSNE")
    if strategy not in valid_strats:
        raise ValueError(f"{strategy} is not a valid dimensionality reduction strategy")
    
    if strategy == valid_strats[0]:
        reducer = cuml.PCA(n_components = n_components)
    elif strategy == valid_strats[1]:
        reducer = cuml.UMAP(n_components = 2)
    elif strategy == valid_strats[2]:
        reducer = cuml.PCA(n_components = 2)
        
    X_hat = reducer.fit_transform(X,y)

    if plot_viz:
        f, ax = plt.subplots(figsize=(10,5))
        scatter = ax.scatter(X_hat[:, 0], X_hat[:, 1], c=y)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        
        cb = plt.colorbar(scatter, spacing='proportional')
        cb.set_label(prefix)
        plt.show()
        
    return X_hat

def open_func(base_path, prefix):
    sets = {}

    for path in base_path.glob(f'{prefix}_*.csv'):
        fname = path.stem
        parts = fname.split('_')
        
        if len(parts) > 2:
            continue
        
        kind = parts[1]

        df = pd.read_csv(path)
        cols = df.columns
        df = df[cols[::-1]]
        df.columns = ['labels', 'sequence']
        
        sets[kind] = df
        
    return sets


def load_dataset(kind = 'train', reduce = False, scale = True, to_torch = False):
    args_dict = {
        'model_name': 'esm1b_t33_650M_UR50S',
        'open_func': open_func,
        'data_path': data_path,
        'gpu': False
    }

    emb_stabilty = EmbeddingProtein1D(**args_dict)
    
    dset = emb_stabilty.generate_datasets('stability', kind = kind, load_embeddings = True)
    
    X, y = dset['embeddings'][:].astype('float32'), dset['labels'][:].reshape(-1,1).astype('float32')
    
    if reduce:
        X = dim_reduction(X, y, plot_viz = False)
        
    if scale: 
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        
        scaler = StandardScaler()
        scaler.fit(y)
        y = scaler.transform(y)
        
    if to_torch:
        X = torch.from_numpy(X)
        y = torch.from_numpy(y)
        
    return X, y