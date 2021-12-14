import unittest

import torch
import utils

class TestUtilsMethods(unittest.TestCase):
    def test_number_of_correct_allcorrect(self):
        #setup
        predictions = [0,1,2,3]
        targets = [0,1,2,3]

        correct, total = utils.number_of_correct(predictions, targets, 4)

        #assert
        expected = [1,1,1,1]
        self.assertEqual(correct, expected)
        self.assertEqual(total, total)

    def test_number_of_correct_somewrong1(self):
        #setup
        predictions = [0,0,1,3]
        targets = [0,1,2,3]

        correct, total = utils.number_of_correct(predictions, targets, 4)

        #assert
        expected_correct = [1,0,0,1]
        expected_total = [1,1,1,1]
        self.assertEqual(correct, expected_correct)
        self.assertEqual(total, expected_total)

    def test_number_of_correct_somewrong2(self):
        #setup
        predictions = [2,3,6,7,0,2,2,6]
        targets =     [6,2,3,7,1,2,2,6]

        correct, total = utils.number_of_correct(predictions, targets, 8)

        #assert
        expected_correct = [0,0,2,0,0,0,1,1]
        expected_total =   [0,1,3,1,0,0,2,1]
        self.assertEqual(correct, expected_correct)
        self.assertEqual(total, expected_total)

    def test_number_of_correct_wrongsizepredic(self):
        predictions = [0,1,2,3]
        targets = [0,1,2,3,4]
        
        self.assertRaises(AssertionError, utils.number_of_correct, predictions, targets, 5)

    def test_number_of_correct_wrongsizetarget(self):
        predictions = [0,1,2,3,4]
        targets = [0,1,2,3]
        
        self.assertRaises(AssertionError, utils.number_of_correct, predictions, targets, 5)

    def test_number_of_correct_tensor(self):
        predictions = torch.tensor([0,1,2,3])
        targets = torch.tensor([0,1,2,3])

        correct, total = utils.number_of_correct(predictions, targets, 4)

        #assert
        expected_correct = [1,1,1,1]
        expected_total   = [1,1,1,1]
        self.assertEqual(correct, expected_correct)
        self.assertEqual(total, expected_total)

    def test_number_of_correct_tensor2(self):
        predictions = torch.tensor([0,1,2,2])
        targets = torch.tensor([0,0,2,2])

        correct, total = utils.number_of_correct(predictions, targets, 4)

        #assert
        expected_correct = [1,0,2,0]
        expected_total   = [2,0,2,0]
        self.assertEqual(correct, expected_correct)
        self.assertEqual(total, expected_total)

if __name__ == '__main__':
    unittest.main()