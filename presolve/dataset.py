import os
from collections.abc import Callable
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from spectral import envi

from spectf.toa import l1b_to_toa_arr
from spectf.utils import drop_bands, envi_header


class ImageDataset(Dataset):
    def __init__(self, sp_paths: list, atm_paths: list, 
                 nchunks: int, chunksize: int, nbins: int, dtype=np.float32):
        self.chunksize = chunksize
        self.nbins = nbins
        self.index= [i for i in range(len(sp_paths) * nchunks)]
        self.dtype = dtype

        self.sp_paths, self.atm_paths = [], []
        for sp_path, atm_path in zip(sp_paths, atm_paths):
            self.sp_paths += [sp_path for i in range(nchunks)]
            self.atm_paths += [atm_path for i in range(nchunks)]

        self.row_cols = []
        for atm_path in self.atm_paths:
            atm = envi.open(envi_header(atm_path))
            h2o_idx = [
                i for i, n in enumerate(atm.metadata['band names']) 
                if n == 'H2O (g cm-2)'
            ]
            atm_im = atm.open_memmap(interleave='bip')
            sampled = False
            while not sampled:
                row = np.random.randint(atm.shape[0] - self.chunksize)
                col = np.random.randint(atm.shape[1] - self.chunksize)
                if np.any(np.isnan(
                    atm[row:row+chunksize, col:col+chunksize, h2o_idx]
                )):
                    continue

                if np.any(
                    atm[row:row+chunksize, col:col+chunksize, h2o_idx]
                    == -9999.
                ):
                    continue

                sampled = True

            self.row_cols.append([row, col])

    @staticmethod
    def calc_histogram(data, nbins, nodata=-9999.):
        data = data[~np.isnan(data)]
        data = data[data != nodata]
        hist_vals, hist_edges = np.histogram(data, bins=nbins, density=True)
        hist_centers = np.maximum(hist_edges[:-1] + np.diff(hist_edges) / 2, 0)

        latent = np.array([hist_centers, hist_vals]).T

        return latent

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rdn = envi.open(envi_header(self.sp_paths[idx])).open_memmap(interleave='bip')
        atm = envi.open(envi_header(self.atm_paths[idx]))

        atm_idx = [
            i for i in range(len(atm.metadata['band names'])) 
            if atm.metadata['band names'][i] == 'H2O (g cm-2)'
        ][0]
        spacecraft_idx = [
            i for i in range(len(atm.metadata['band names'])) 
            if atm.metadata['band names'][i] == 'Spacecraft Flag'
        ][0]
        atm = atm.open_memmap(interleave='bip')

        row, col = self.row_cols[idx]

        sample = atm[row:row+self.chunksize, col:col+self.chunksize, :]
        bad_rows, bad_cols = np.where(sample[..., spacecraft_idx])
        sample = sample[..., atm_idx].copy()
        sample[bad_rows, bad_cols] = np.nan

        latent = np.moveaxis(
            self.calc_histogram(sample, self.nbins),
            -1, 0
        )

        # How to handle NaN rdn? -> Fill with chunk median.
        rdn = rdn[row:row+self.chunksize, col:col+self.chunksize, :].copy()
        # rdn[bad_rows, bad_cols, :] = np.nanmedian(rdn, axis=(0, 1))

        rdn = np.moveaxis(
            rdn,
            -1, 0
        )

        return {
            'images': rdn.astype(self.dtype),
            'latent': latent.astype(self.dtype)
        }
