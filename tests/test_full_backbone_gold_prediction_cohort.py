"""Exercise the actual prediction helper before expensive MRI admission."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / 'scripts/ticket64_full_backbone_recovery_v2.py'


class PredictionAdmissionReached(Exception):
    pass


def prediction_function(source, manifest, expected):
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'final_predictions')
    function = ast.Module(body=[node], type_ignores=[])

    def load_bank(plan, selected):
        if selected != expected:
            raise ValueError('Selected source cohort incomplete')
        raise PredictionAdmissionReached

    shared = SimpleNamespace(load_fold_manifest=lambda _: manifest,
                             t36=SimpleNamespace(FOLD_MANIFEST_PATH=None), load_bank=load_bank)
    namespace = dict(original=shared, require=lambda c, m: None)
    exec(compile(function, str(source), 'exec'), namespace)
    return namespace['final_predictions']


def test_final_prediction_admits_only_gold_from_a_mixed_heldout_fold():
    manifest = SimpleNamespace(
        held_out_study_uids=lambda _: ('gold-held', 'report-held-no-pixels'),
        gold_memberships=(SimpleNamespace(study_instance_uid='gold-held', fold=0),
                          SimpleNamespace(study_instance_uid='gold-training', fold=1)),
    )
    helper = prediction_function(SOURCE, manifest, {'gold-held'})
    with pytest.raises(PredictionAdmissionReached):
        helper(None, {}, 0, 'full-backbone')


def test_original_mixed_cohort_failure_remains_preserved():
    source = PROJECT / 'scripts/ticket64_full_backbone_comparison.py'
    manifest = SimpleNamespace(
        held_out_study_uids=lambda _: ('gold-held', 'report-held-no-pixels'),
        gold_memberships=(SimpleNamespace(study_instance_uid='gold-held', fold=0),),
    )
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'final_predictions')
    def load_bank(plan, selected):
        if selected != {'gold-held'}:
            raise ValueError('Selected source cohort incomplete')
        raise PredictionAdmissionReached
    namespace = dict(load_fold_manifest=lambda _: manifest,
                     t36=SimpleNamespace(FOLD_MANIFEST_PATH=None), load_bank=load_bank)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    with pytest.raises(ValueError, match='Selected source cohort incomplete'):
        namespace['final_predictions'](None, {}, 0, 'full-backbone')
