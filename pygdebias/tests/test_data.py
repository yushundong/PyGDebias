# -*- coding: utf-8 -*-
import unittest
from numpy.testing import assert_raises

from pygdebias.datasets import (
    Google,
    Facebook,
    Oklahoma,
    UNC,
    Twitter,
    Lastfm,
    Nba,
    Ml_1m,
    Ml_20m,
    Ml_100k,
    German,
    dataset,
    Credit,
    Cora,
    Pokec_n,
    Pokec_z,
    Filmtrust,
    Citeseer,
    Yelp,
    Bail,
    Amazon,
    LCC,
    LCC_small,
    Epinion,
    Ciao,
    Dblp,
    German,
)


class Test(unittest.TestCase):
    def test_sum(self):
        self.assertEqual(1, 1)

    def test_google(self) -> None:
        dataset = Google()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

        # facebook = Facebook()
        # oklahoma = Oklahoma()
        # unc = UNC()
        # twitter = Twitter()
        # lastfm = Lastfm()
        # nba = Nba()
        # ml_1m = Ml_1m()
        # ml_20m = Ml_20m()
        # ml_100k = Ml_100k()
        # german = German()
        # dataset = dataset()
        # credit = Credit()
        # cora = Cora()
        # pokec_n = Pokec_n()
        # pokec_z = Pokec_z()
        # filmtrust = Filmtrust()
        # citeseer = Citeseer()
        # yelp = Yelp()
        # amazon = Amazon()
        # lcc = LCC()
        # lcc_small = LCC_small()
        # epinion = Epinion()
        # ciao = Ciao()
        # dblp = Dblp()
        # german = German()

    def test_facebook(self) -> None:
        dataset = Facebook()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

    def test_nba(self) -> None:
        dataset = Nba()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

    def test_pokec_z(self) -> None:
        dataset = Pokec_z()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

    def test_pokec_n(self) -> None:
        dataset = Pokec_n()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

    def test_twitter(self) -> None:
        dataset = Twitter()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

    def test_facebook(self) -> None:
        dataset = Facebook()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

    def test_cora(self) -> None:
        dataset = Cora()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

    def test_Citeseer(self) -> None:
        dataset = Citeseer()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

    def test_german(self) -> None:
        dataset = German()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

    def test_bail(self) -> None:
        dataset = Bail()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

    def test_credit(self) -> None:
        dataset = Credit()
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
            dataset.adj(),
            dataset.features(),
            dataset.idx_train(),
            dataset.idx_val(),
            dataset.idx_test(),
            dataset.labels(),
            dataset.sens(),
            dataset.sens_idx(),
        )

    def test_others(self) -> None:
        dataset = Lastfm()
        dataset = Ml_1m()
        dataset = Ml_20m()
        dataset = Ml_100k()
        dataset = Yelp()
        dataset = Amazon()
        dataset = LCC()
        dataset = LCC_small()
        dataset = Epinion()
        dataset = Ciao()
        dataset = Dblp()
        dataset = UNC()
        dataset = Oklahoma()
        dataset = Filmtrust()


if __name__ == "__main__":
    unittest.main()
