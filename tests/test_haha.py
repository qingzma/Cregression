from unittest import TestCase
from cpm.Data1 import haha

class TestHaha(TestCase):
    def test_something(self):
        ha=haha()
        self.assertEqual(ha.f(2),2)

