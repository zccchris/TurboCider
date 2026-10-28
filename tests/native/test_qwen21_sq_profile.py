import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('sq_profile', ROOT / 'tools/native/embed_qwen21_sq_profile.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)

class FrozenProfileTests(unittest.TestCase):
    def test_frozen_profile_embeds_without_external_paths(self):
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / 'embedded.hpp'
            profile.generate(profile.ASSETS, output)
            source = output.read_text()
            self.assertIn('inline constexpr uint16_t inverse_bits[]', source)
            self.assertIn('777c4179010b3f662515013bbd51e1e57e792b9635b3d55a9fea073956a9eedc', source)
            self.assertNotIn('/Users/', source)

    def test_corrupt_payload_and_index_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            shutil.copytree(profile.ASSETS, directory / 'asset')
            asset = directory / 'asset'
            payload = asset / 'sq_convrot_v1.f16le'
            original = payload.read_bytes()
            payload.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, 'integrity'):
                profile.generate(asset, directory / 'out.hpp')
            payload.write_bytes(original)
            index = asset / 'sq_convrot_v1.json'
            data = json.loads(index.read_text())
            data['entries'][0]['offset'] = 1
            index.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'geometry'):
                profile.generate(asset, directory / 'out.hpp')

    def test_editing_profile_is_independent_and_integrity_checked(self):
        edit = json.loads((profile.ASSETS / 'sq_convrot_edit_v1.json').read_text())
        generation = json.loads((profile.ASSETS / 'sq_convrot_v1.json').read_text())
        self.assertNotEqual(edit['identity'], generation['identity'])
        self.assertNotEqual(edit['payload_sha256'], generation['payload_sha256'])
        self.assertEqual(edit['source_sha256'], generation['source_sha256'])
        self.assertEqual(edit['operation'], 'image.edit')
        self.assertEqual([len(r['reference_sha256']) for r in edit['calibration_requests']], [1, 2, 3])
        with tempfile.TemporaryDirectory() as raw:
            asset = Path(raw) / 'asset'
            shutil.copytree(profile.ASSETS, asset)
            output = Path(raw) / 'edit.hpp'
            profile.generate(asset, output, stem='sq_convrot_edit_v1', namespace='sq_edit_profile')
            self.assertIn('namespace tc::qwen21::sq_edit_profile', output.read_text())
            self.assertNotIn('/Users/', output.read_text())
            payload = asset / 'sq_convrot_edit_v1.f16le'
            payload.write_bytes(payload.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, 'integrity'):
                profile.generate(asset, output, stem='sq_convrot_edit_v1', namespace='sq_edit_profile')

    def test_runtime_identity_binds_payload_and_mapping(self):
        spec = importlib.util.spec_from_file_location('identity', ROOT / 'tools/native/generate_runtime_build_identity.py')
        identity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(identity)
        paths = {entry['path'] for entry in identity.source_inputs(ROOT)}
        for name in ('sq_convrot_v1.json', 'sq_convrot_v1.f16le', 'sq_convrot_edit_v1.json', 'sq_convrot_edit_v1.f16le'):
            self.assertIn('native/models/qwen21/calibration/' + name, paths)

if __name__ == '__main__':
    unittest.main()
