import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

API_DIR = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('ids_api_under_test', API_DIR / 'app.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class PredictionApiTest(unittest.TestCase):
    def setUp(self):
        self.client = api.app.test_client()
        self.log = patch.object(api, 'append_log').start()
        self.forward = patch.object(api, 'forward_to_laravel').start()
        self.addCleanup(patch.stopall)

    def test_selected_model_is_identifiable(self):
        response = self.client.get('/api/features')
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertEqual(result['count'], 78)
        self.assertEqual(result['model_filename'], 'cicids78_demo.joblib')
        self.assertEqual(result['model_sha256'], '73893343b1d6f8b84b86dba6dc66e61f5a456d1472aa10859310aa04ff9e982e')

    def test_incomplete_flow_is_rejected_without_forwarding(self):
        response = self.client.post('/api/analyze', json={'flow_duration': 10})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()['error'], 'missing_model_features')
        self.log.assert_not_called()
        self.forward.assert_not_called()

    def test_invalid_json_and_numeric_features_are_rejected(self):
        self.assertEqual(self.client.post('/api/analyze', json=[]).status_code, 400)
        flow = dict.fromkeys(api.feature_names, 0)
        for invalid in [None, 'not a number', 'NaN', 'Infinity']:
            flow['flow_duration'] = invalid
            self.assertEqual(self.client.post('/api/analyze', json=flow).status_code, 422)
        self.forward.assert_not_called()

    def test_all_current_demo_fixtures_through_api(self):
        fixtures_path = API_DIR.parent / 'resources/data/demo-attacks.json'
        if not fixtures_path.exists():
            self.skipTest('Parent application demo fixtures are not available in this checkout')
        fixtures = json.loads(fixtures_path.read_text())
        total = benign = correct = 0
        normalize = lambda text: ''.join(c for c in text.upper() if c.isalnum())
        for selected, samples in fixtures.items():
            for sample in samples:
                response = self.client.post('/api/analyze', json=sample)
                self.assertEqual(response.status_code, 200)
                result = response.get_json()
                self.assertEqual(result['missing_expected_features'], 0)
                total += 1
                benign += result['risk_level'] == 'benign'
                correct += normalize(result['attack_type'] or '') == normalize(selected)
        self.assertEqual((total, benign, correct), (70, 0, 65))
        self.assertEqual(self.forward.call_count, 70)


if __name__ == '__main__':
    unittest.main()
