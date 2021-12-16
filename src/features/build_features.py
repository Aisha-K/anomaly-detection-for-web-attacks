import os
import logging
import pandas as pd, numpy as np
import regex as re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

COMMON_SQL_WORDS = ['create', 'insert', 'view', 'from' , 'select', 'alter', 'add', 'distinct', 'into','update','set','delete',
                    'truncate','as','order','between','where','and','or','null','drop','column','table','database','group',
                    'having','join','union','exists','like','case']

COMMON_BATCH_WORDS = ["cd", "ls", "cat", "cd", "sudo", "tail", "echo", "grep", "mv", "less","more","gnome-open",
                     "chmod","chown","chgrp","find", "wget","curl", "su"]

'''
returns the number of words in x that are in words_list
'''
def num_in_words(x, words_list):
    count = 0
    w = x.lower()
    for w in words_list:
        if w in x:
            count +=1
    return count

def num_special_chars(x):
    y = ''.join(x)
    z = re.sub(r'[\w]+','',y)
    return len(z)


def extract_url_only(x):
    if '?' in x:
        return x.split('?')[0]
    return x[:-8]

def extract_args_only(row):
    if row['method'] == 'GET':
        x = row['url']
        if '?' in x:
            return re.split( '[&=]', x.split('?')[1] )
        return []    
    elif (row['method'] == 'POST') | (row['method'] == 'PUT'):
        x = row['body']
        if type(x)==str:
            return re.split('[&=]', x)
        return []        

#returns a df containing the document vectors as columns    
def vector_features(df, vec_size):
    card_docs = [TaggedDocument(words, [i]) 
            for i, words in enumerate(df.arg_words)]
    #create vector model
    model = Doc2Vec(vector_size=vec_size, min_count=1, epochs = 30)
    model.build_vocab(card_docs)
    model.train(card_docs, total_examples=model.corpus_count, epochs=model.epochs)
    card2vec = [model.infer_vector(df['arg_words'][i]) 
                for i in range(0,len(df['arg_words']))]
    dtv= np.array(card2vec).tolist()
    
    # add vector to df as columns
    df_vecs = pd.DataFrame( dtv )
    df_vecs['anomalous'] = df['anomalous']
    return df_vecs

def main():
    logger = logging.getLogger(__name__)

    # load data
    file_dir = os.path.dirname(__file__)
    df = pd.read_csv(  os.path.join(file_dir,'..','..','data', 'interim','1_original_data_to_df.csv') , index_col = 0)
    
    #processing the columns
    df['browser'] = df['User-Agent'].str.extract( r'^(.*?) \(', expand=False)
    df['system-information'] = df['User-Agent'].str.extract( r'\((.*?)\)', expand=False)
    df['platform'] = df['User-Agent'].str.extract( r'\) (.*)$', expand=False)
    df.drop('User-Agent',1)
    df['protocol'] = df['url'].str.extract(r' (.*?)$')
    df['url_only'] = df['url'].apply(lambda x: extract_url_only(x))
    df['url_words'] = df['url'].apply(lambda x:  re.split('[/]', x) )
    df['arg_words'] = df.apply(lambda x: extract_args_only(x), axis=1)

    ## extracting more feature characteristics
    df['num_of_args'] = df['arg_words'].apply(lambda x: len(x))
    df['max_length_of_args'] = df['arg_words'].apply(lambda x: 0 if len(x) ==0 else max([ len(i) for i in x ] ))
    df['min_length_of_args'] = df['arg_words'].apply(lambda x: 0 if len(x) ==0 else min([ len(i) for i in x ] ))
    df['total_length_args'] = df['arg_words'].apply(lambda x: sum( [ len(i) for i in x ] ))

    df['total_length_request'] = df['url'].apply(lambda x: len(x))
    df['lenght_of_path'] = df['url_only'].apply(lambda x: len(x))
    df['port_is_common'] = df['Host'].apply(lambda x: x.split(':')[-1] in ['80','443','8080'] )

    df['num_of_paths'] = df['url_words'].apply(lambda x: len(x) )
    df['num_sql_words'] = df['url'].apply(lambda x: num_in_words(x, COMMON_SQL_WORDS))
    df['num_batch_words'] = df['url'].apply(lambda x: num_in_words(x, COMMON_BATCH_WORDS))
    df['num_special_chars'] = df['arg_words'].apply(lambda x: num_special_chars(x))


    df = df.drop(['body','url_only','Accept','Pragma','Cache-control', 'url', 'User-Agent','Cookie','Accept-Language', 'Accept-Encoding', 'Accept-Charset','Connection'], 1)


    #one hot encoding
    to_encode_one_hot = ['method', 'Host','Content-Type', 'browser', 'platform','protocol', 'system-information']
    enc = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
    enc.fit(df[to_encode_one_hot] )
    encoded = pd.DataFrame( enc.transform( df[to_encode_one_hot] ), columns=enc.get_feature_names() )
    df = pd.concat ([df, encoded] , axis=1)

    #fill nulls
    df['Content-Length'] = df['Content-Length'].fillna(0)
    df['Content-Length'] = df['Content-Length'].astype(int)

    df = df.drop(to_encode_one_hot, axis=1)

    df.drop(['url_words', 'arg_words'], axis=1).to_csv(os.path.join(file_dir,'..','..','data','processed','features_with_doc_vector_0.csv'))
    
    logger.info("Starting arguments to doc vectorization:")
    #vectorization of diferent sizes, save each dataframe created from those vectors
    for i in [30, 60, 90, 120]:
        logger.info(i)
        df_vecs = vector_features(df, i)
        df_combined = df.drop(['anomalous', 'url_words', 'arg_words'], axis=1).join(df_vecs)
        df_combined.to_csv(os.path.join(file_dir,'..','..','data','processed',f'features_with_doc_vector_{i}.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
    