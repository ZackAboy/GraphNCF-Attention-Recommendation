import zipfile
from torch.utils.data import Dataset
import torch
import os
import pandas  as pd
import numpy as np
from zipfile import ZipFile
import requests
import sklearn
import random

class Download():
    def __init__(self,
                 root: str,
                 file_size: str = '100k',
                 download: bool = True,
                 ) -> None:
        self.root = root
        self.download = download
        self.file_size_url = file_size
        if self.file_size_url == '100k' or self.file_size_url == '20m':
            self.file_url = 'ml-latest' if self.file_size_url == '20m' else 'ml-latest-small'
            self.fname = os.path.join(self.root, self.file_url, 'ratings.csv')
        else:
            if self.file_size_url == '10m':
                self.file_url = 'ml-' + self.file_size_url
                self.extracted_file_dir = 'ml-10M100K'
            if self.file_size_url =='1m':
                self.file_url = 'ml-' + self.file_size_url

            if self.file_size_url=='10m':
                self.fname = os.path.join(self.root, self.extracted_file_dir, 'ratings.dat')
            else:
                self.fname = os.path.join(self.root, self.file_url, 'ratings.dat')

        if self.download or not os.path.isfile(self.fname):
            self._download_movielens()
        self.df = self._read_ratings_csv()

    def _download_movielens(self) -> None:
        file = self.file_url + '.zip'
        url = ("http://files.grouplens.org/datasets/movielens/" + file)
        req = requests.get(url, stream=True)
        print('Downloading MovieLens dataset')
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        with open(os.path.join(self.root, file), mode='wb') as fd:
            for chunk in req.iter_content(chunk_size=None):
                fd.write(chunk)
        with ZipFile(os.path.join(self.root, file), "r") as zip:
            # Extract files
            print("Extracting all the files now...")
            zip.extractall(path=self.root)
            print("Downloading Complete!")

    def _read_ratings_csv(self) -> pd.DataFrame:
        print("Reading file")
        if not os.path.isfile(self.fname):
            self._download_movielens()

        if self.file_size_url == '100k' or self.file_size_url == '20m':
            df = pd.read_csv(self.fname, sep=',')
        else:
            df = pd.read_csv(self.fname, sep="::", header=None,
                               names=['userId', 'movieId', 'ratings', 'timestamp'])
        df = df.drop(columns=['timestamp'])
        print("Reading Complete!")
        return df

    def split_train_test(self) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        train_dataframe = self.df
        test_dataframe = train_dataframe.sample(frac=1).drop_duplicates(['userId'])
        train_dataframe = pd.concat([train_dataframe, test_dataframe])
        train_dataframe = train_dataframe.drop_duplicates(keep=False)


        train_dataframe.loc[:, 'rating'] = 1
        test_dataframe.loc[:, 'rating'] = 1

        test_dataframe = test_dataframe.sort_values(by=['userId'],axis=0)
        print(f"len(total): {len(self.df)}, len(train): {len(train_dataframe)}, len(test): {len(test_dataframe)}")
        return self.df, train_dataframe, test_dataframe,


class MovieLens(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 total_df: pd.DataFrame,
                 ng_ratio: int,
                 train:bool=False,
                 )->None:
        super(MovieLens, self).__init__()

        self.df = df
        self.total_df = total_df
        self.train = train
        self.ng_ratio = ng_ratio
        self.users, self.items = self._negative_sampling()
        print(f'len items:{self.items.shape}')

    def __len__(self) -> int:
        return len(self.users)


    def __getitem__(self, index):
        if self.train:
            return self.users[index], self.items[index][0], self.items[index][1]
        else:
            return self.users[index], self.items[index]


    def _negative_sampling(self) :
        df = self.df
        total_df = self.total_df
        users, items = [], []
        user_item_set = set(zip(df['userId'], df['movieId']))
        total_user_item_set = set(zip(total_df['userId'],total_df['movieId']))
        all_movieIds = total_df['movieId'].unique()
        for u, i in user_item_set:
            visit = []
            item = []
            if not self.train:
                items.append(i)
                users.append(u)
            else:
                item.append(i)

            for k in range(self.ng_ratio):
                negative_item = np.random.choice(all_movieIds)
                while (u, negative_item) in total_user_item_set or negative_item in visit:
                    negative_item = np.random.choice(all_movieIds)

                if self.train:
                    item.append(negative_item)
                    visit.append(negative_item)
                else:
                    items.append(negative_item)
                    visit.append(negative_item)
                    users.append(u)

            if self.train:
                items.append(item)
                users.append(u)

        return torch.tensor(users), torch.tensor(items)
