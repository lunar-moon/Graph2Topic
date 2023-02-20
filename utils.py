from octis.dataset.dataset import Dataset

def prepare_dataset(dataset_name):
    
    dataset = Dataset()
    
    if dataset_name == '20ng':
        dataset.load_custom_dataset_from_folder("./data/20NewsGroup")
    elif dataset_name == 'bbc':
        dataset.load_custom_dataset_from_folder("./data/BBC_News")
    elif dataset_name == 'm10':
        dataset.load_custom_dataset_from_folder("./data/M10")
    elif dataset_name == 'beer':
        dataset.load_custom_dataset_from_folder("./data/beer")
    elif dataset_name == 'crr':
        dataset.load_custom_dataset_from_folder("./data/crr")
    elif dataset_name == 'asap':
        dataset.load_custom_dataset_from_folder("./data/asap")
    elif dataset_name == 'nlpcc':
        dataset.load_custom_dataset_from_folder("./data/nlpcc")
    elif dataset_name == 'nlpcc_c':
        dataset.load_custom_dataset_from_folder("./data/nlpcc_c")
        
    # make sentences and token_lists
    token_lists = dataset.get_corpus()
    sentences = [' '.join(text_list) for text_list in token_lists]
    
    return dataset, sentences
