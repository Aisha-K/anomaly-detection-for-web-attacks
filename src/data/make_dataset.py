# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import regex as re


'''
Outputs an array of dictionaries, with each dictionary representing a request.
@param data_array an array containing each line of the txt file as an item
'''
def process_http_data(data_array, is_anomalous):
    data_processed = []
    i = 0 
    while i < (len(data_array)-1):
        data_item = {}
        if data_array[i][0:3] == 'GET':
            data_item['method']='GET'
            data_item['url']= data_array[i][4:-1]
            i+=1

            while (i<len(data_array)-1) & (data_array[i]!='\n') :
                s = data_array[i].split(':',1)
                data_item[ s[0] ] = s[1][1:-1]
                i += 1
            i +=2

        elif (data_array[i][0:4] == 'POST') | (data_array[i][0:3] == 'PUT'):
            if (data_array[i][0:3] == 'PUT'):
                data_item['method']='PUT'
            else:
                data_item['method']='POST'
            data_item['url']= data_array[i][5:-1]
            i+=1

            while data_array[i]!='\n':
                s = data_array[i].split(':',1)
                data_item[s[0] ] = s[1][1:-1]
                i += 1

            i += 1
            data_item['body'] = data_array[i][0:-1]
            i+= 2

        else:
            i+=1
            continue

        data_item['anomalous'] = is_anomalous
        data_processed.append(data_item)

    return data_processed


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


    #reading in data:
    with open('../../data/raw/normalTrafficTest.txt') as file:
        data1 =  file.readlines()

    with open('../../data/raw/normalTrafficTraining.txt') as file:
        data2 = file.readlines()

    with open('../../data/raw/anomalousTrafficTest.txt') as file:
        data3 = file.readlines()

    d3 = process_http_data(data3, True)
    d2 = process_http_data(data2, False)
    d1 = process_http_data(data1, False)


    logger.info(len(df) )

    df_orig = pd.DataFrame(d1 + d2 + d3)
    del(d3,d2,d1)

    df.to_csv('../../data/interim/1_original_data_to_df.csv')
            


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
