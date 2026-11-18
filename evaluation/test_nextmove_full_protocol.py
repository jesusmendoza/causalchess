"""Protect the evaluation boundaries that determine scientific validity."""
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import chess
import chess.pgn
import numpy as np
import torch

from nextmove_full_vocab import (process_game, frequency_baseline, group_splits,
                                 validate_splits, classifier_metrics, encoder,
                                 TransferModel, MODELS)
from prev_move_models import create_model


class ProtocolTests(unittest.TestCase):
    def test_target_ply_window_and_temporal_direction(self):
        game = chess.pgn.read_game(io.StringIO('1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 *'))
        pairs = process_game(game, min_ply=6, max_ply=8)
        self.assertEqual([move for _,move in pairs], ['a7a6','b5a4','g8f6'])
        for target_ply,(fen,move) in enumerate(pairs,6):
            board=chess.Board(fen)
            self.assertEqual(board.ply(), target_ply-1)
            self.assertIn(chess.Move.from_uci(move), board.legal_moves)

    def test_majority_uses_training_labels_not_test_frequency(self):
        result=frequency_baseline(np.array([1,1,1,0]),np.array([0,0,0,1]),6)
        self.assertEqual(result['selected_classes'][0],1)
        self.assertEqual(result['top1'],.25)

    def test_whole_games_and_duplicate_sequences_remain_together(self):
        groups=np.repeat(np.arange(100),10)
        split=group_splits(groups,0,{'train':200,'val':50,'test':60})
        validate_splits(groups,split)
        again=group_splits(groups,0,{'train':200,'val':50,'test':60})
        for name in split: np.testing.assert_array_equal(split[name],again[name])
        bad={'train':np.array([0]),'test':np.array([1])}
        with self.assertRaises(AssertionError): validate_splits(groups,bad)

    def test_unseen_test_class_counts_as_error_without_dropping_row(self):
        class Dummy:
            classes_=np.array([1,7])
            def predict_proba(self,X): return np.array([[.8,.2],[.8,.2],[.8,.2]])
        metrics=classifier_metrics(Dummy(),None,np.array([1,7,99]))
        self.assertAlmostEqual(metrics['top1'],1/3)
        self.assertAlmostEqual(metrics['top5'],2/3)

    def test_reproducible_scratch_and_paired_fresh_head(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with TemporaryDirectory() as folder, torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            config = {'arch': 'cnn', 'n_moves': 12, 'embed_dim': 256}
            reference = create_model(**config)
            checkpoint = Path(folder) / 'fixture.pt'
            torch.save({**config, 'model': reference.state_dict()}, checkpoint)
            with patch.dict(MODELS, {'PMP-mem': checkpoint}):
                first = encoder('Scratch', 0)
                second = encoder('Scratch', 0)
                loaded = encoder('PMP-mem', 0)
            for left, right in zip(first.parameters(), second.parameters()):
                self.assertTrue(torch.equal(left, right))
            for name, parameter in loaded.named_parameters():
                self.assertTrue(torch.equal(parameter, reference.state_dict()[name]))
            self.assertFalse(torch.equal(next(first.parameters()), next(loaded.parameters())))
            scratch = TransferModel(first, 1932, 0)
            pretrained = TransferModel(loaded, 1932, 0)
            self.assertTrue(torch.equal(scratch.head.weight, pretrained.head.weight))
            self.assertTrue(torch.equal(scratch.head.bias, pretrained.head.bias))


if __name__=='__main__': unittest.main()
