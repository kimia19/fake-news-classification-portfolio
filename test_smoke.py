"""Offline tests for independently rewritten FA-KES portfolio modules."""
import unittest
import numpy as np
from neural_experiment import clean,make_network
from split_fakes import make_group_split


class PortfolioSmokeTests(unittest.TestCase):
    def test_clean_title(self):
        self.assertEqual(clean("  FAKE, News!  "),"fake news")

    def test_disjoint_title_groups(self):
        labels=np.array([i%2 for i in range(200)])
        titles=[f"Headline {i//2}" for i in range(200)]
        a,b,c,groups=make_group_split(titles,labels,seed=42)
        self.assertEqual(len(a)+len(b)+len(c),200)
        self.assertFalse(set(groups[a])&set(groups[c]))

    def test_keras_architectures_forward(self):
        matrix=np.zeros((5000,100),dtype=np.float32)
        for name in ("LSTM","CNN-LSTM","Regularized CNN-LSTM"):
            model=make_network(name,matrix,42)
            x=np.zeros((2,40),dtype=np.int32)
            y=model(x,training=False).numpy()
            self.assertEqual(y.shape,(2,1))
            self.assertTrue(np.isfinite(y).all())


if __name__=="__main__":
    unittest.main()
