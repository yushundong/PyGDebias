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
    Bail,
    Credit,
    Cora,
    Pokec_n,
    Pokec_z,
    Filmtrust,
    Citeseer,
    Yelp,
    Amazon,
    LCC,
    LCC_small,
    Epinion,
    Ciao,
    Dblp,
    German,
)


class Test(unittest.TestCase):
    def test_initialize(self) -> None:
        with assert_raises(ValueError):
            pass
            # self.google = Google()
            # self.facebook = Facebook()
            # self.oklahoma = Oklahoma()
            # self.unc = UNC()
            # self.twitter = Twitter()
            # self.lastfm = Lastfm()
            # self.nba = Nba()
            # self.ml_1m = Ml_1m()
            # self.ml_20m = Ml_20m()
            # self.ml_100k = Ml_100k()
            # self.german = German()
            # self.bail = Bail()
            # self.credit = Credit()
            # self.cora = Cora()
            # self.pokec_n = Pokec_n()
            # self.pokec_z = Pokec_z()
            # self.filmtrust = Filmtrust()
            # self.citeseer = Citeseer()
            # self.yelp = Yelp()
            # self.amazon = Amazon()
            # self.lcc = LCC()
            # self.lcc_small = LCC_small()
            # self.epinion = Epinion()
            # self.ciao = Ciao()
            # self.dblp = Dblp()
            # self.german = German()

    def test_api(self):
        pass
        # adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx = (
        #     self.bail.adj(),
        #     self.bail.features(),
        #     self.bail.idx_train(),
        #     self.bail.idx_val(),
        #     self.bail.idx_test(),
        #     self.bail.labels(),
        #     self.bail.sens(),
        #     self.bail.sens_idx(),
        # )
        pass


if __name__ == "__main__":
    unittest.main(argv=[""])
