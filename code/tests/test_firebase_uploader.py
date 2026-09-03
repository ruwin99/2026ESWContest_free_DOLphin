from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from firebase_uploader import (  # noqa: E402
    FirebaseUploadConfig,
    FirebaseUploadError,
    load_firebase_upload_config,
    upload_dashboard_manifest,
)


class FirebaseUploaderTests(unittest.TestCase):
    def test_missing_config_leaves_upload_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing.env"
            config = load_firebase_upload_config(
                environment={"RAIL_ROBOT_FIREBASE_CONFIG_FILE": str(missing)}
            )

        self.assertIsNone(config)

    def test_environment_loads_pinned_device_identity(self) -> None:
        config = load_firebase_upload_config(
            environment={
                "RAIL_ROBOT_FIREBASE_UPLOAD": "1",
                "RAIL_ROBOT_FIREBASE_API_KEY": "example-public-web-key",
                "RAIL_ROBOT_FIREBASE_UID": "jetson-device-uid",
                "RAIL_ROBOT_FIREBASE_EMAIL": "device@example.invalid",
                "RAIL_ROBOT_FIREBASE_PASSWORD": "example-private-password",
            }
        )

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.expected_uid, "jetson-device-uid")
        self.assertEqual(config.project_id, "eswcontest-rail-robot")
        self.assertEqual(
            config.storage_bucket,
            "eswcontest-rail-robot.firebasestorage.app",
        )
        rendered = repr(config)
        self.assertNotIn("example-public-web-key", rendered)
        self.assertNotIn("device@example.invalid", rendered)
        self.assertNotIn("example-private-password", rendered)

    def test_uploads_artifacts_then_manifest_then_firestore_document(self) -> None:
        run_id = "run_" + "a" * 24
        capture_id = "cap_" + "b" * 32
        calls: list[tuple[str, str, dict[str, str], bytes, str]] = []

        def fake_request(
            url: str,
            method: str,
            headers: dict[str, str],
            body: bytes,
            _timeout_seconds: float,
            operation: str,
        ) -> dict:
            calls.append((url, method, headers, body, operation))
            if "signInWithPassword" in url:
                return {"idToken": "id-token", "localId": "jetson-device-uid"}
            return {}

        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = Path(temporary_directory) / "outputs"
            media = outputs / "dashboard" / "media" / run_id
            runs = outputs / "dashboard" / "runs"
            media.mkdir(parents=True)
            runs.mkdir(parents=True)
            artifact_path = media / f"{capture_id}_crack.png"
            artifact_payload = b"\x89PNG\r\n\x1a\npng-payload"
            artifact_path.write_bytes(artifact_payload)
            object_key = f"dashboard/media/{run_id}/{capture_id}_crack.png"
            manifest = {
                "schema_version": 1,
                "exported_at_utc": "2026-09-03T00:00:00Z",
                "runs": [{"run_id": run_id, "status": "complete"}],
                "model_provenance": [],
                "captures": [],
                "analyses": [],
                "artifacts": [
                    {
                        "artifact_id": f"{capture_id}:crack_preview",
                        "capture_id": capture_id,
                        "artifact_type": "crack_preview",
                        "object_key": object_key,
                        "public_url": None,
                        "media_type": "image/png",
                        "width": 2,
                        "height": 2,
                        "sha256": hashlib.sha256(artifact_payload).hexdigest(),
                    }
                ],
                "run_summaries": [],
            }
            manifest_path = runs / "demo.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
            )
            config = FirebaseUploadConfig(
                api_key="example-public-web-key",
                expected_uid="jetson-device-uid",
                email="device@example.invalid",
                password="example-private-password",
            )

            result = upload_dashboard_manifest(
                manifest_path,
                config,
                request_function=fake_request,
            )

        self.assertEqual(result.run_id, run_id)
        self.assertEqual(result.artifact_count, 1)
        self.assertEqual(
            result.uploaded_bytes,
            len(artifact_payload) + len(json.dumps(manifest, ensure_ascii=False).encode("utf-8")),
        )
        self.assertEqual(len(calls), 4)
        self.assertIn("signInWithPassword", calls[0][0])
        self.assertIn(f"dashboard%2Fmedia%2F{run_id}", calls[1][0])
        self.assertEqual(calls[1][2]["Authorization"], "Firebase id-token")
        self.assertIn(artifact_payload, calls[1][3])
        self.assertIn(b'"uploaderUid":"jetson-device-uid"', calls[1][3])
        self.assertIn(f"dashboard%2Fmanifests%2F{run_id}.json", calls[2][0])
        self.assertIn(b'"uploaderUid":"jetson-device-uid"', calls[2][3])
        self.assertEqual(calls[3][1], "PATCH")
        self.assertEqual(calls[3][2]["Authorization"], "Bearer id-token")
        firestore_body = json.loads(calls[3][3].decode("utf-8"))
        fields = firestore_body["fields"]
        self.assertEqual(fields["source"], {"stringValue": "jetson-runtime"})
        self.assertEqual(
            fields["uploader_uid"], {"stringValue": "jetson-device-uid"}
        )
        self.assertEqual(
            fields["payload"]["mapValue"]["fields"]["runs"]["arrayValue"]
            ["values"][0]["mapValue"]["fields"]["run_id"],
            {"stringValue": run_id},
        )

    def test_rejects_invalid_export_ids_before_authentication(self) -> None:
        calls: list[str] = []

        def fake_request(
            url: str,
            _method: str,
            _headers: dict[str, str],
            _body: bytes,
            _timeout_seconds: float,
            _operation: str,
        ) -> dict:
            calls.append(url)
            return {}

        with tempfile.TemporaryDirectory() as temporary_directory:
            runs = Path(temporary_directory) / "outputs" / "dashboard" / "runs"
            runs.mkdir(parents=True)
            manifest_path = runs / "demo.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "runs": [{"run_id": "run_demo", "status": "complete"}],
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            config = FirebaseUploadConfig(
                api_key="example-public-web-key",
                expected_uid="jetson-device-uid",
                email="device@example.invalid",
                password="example-private-password",
            )

            with self.assertRaisesRegex(FirebaseUploadError, "run_id"):
                upload_dashboard_manifest(
                    manifest_path,
                    config,
                    request_function=fake_request,
                )

        self.assertEqual(calls, [])

    def test_rejects_invalid_capture_id_before_authentication(self) -> None:
        run_id = "run_" + "a" * 24
        calls: list[str] = []

        def fake_request(
            url: str,
            _method: str,
            _headers: dict[str, str],
            _body: bytes,
            _timeout_seconds: float,
            _operation: str,
        ) -> dict:
            calls.append(url)
            return {}

        with tempfile.TemporaryDirectory() as temporary_directory:
            runs = Path(temporary_directory) / "outputs" / "dashboard" / "runs"
            runs.mkdir(parents=True)
            manifest_path = runs / "demo.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "runs": [{"run_id": run_id, "status": "complete"}],
                        "artifacts": [
                            {
                                "capture_id": "cap_demo",
                                "artifact_type": "crack_preview",
                                "object_key": (
                                    f"dashboard/media/{run_id}/cap_demo_crack.png"
                                ),
                                "media_type": "image/png",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = FirebaseUploadConfig(
                api_key="example-public-web-key",
                expected_uid="jetson-device-uid",
                email="device@example.invalid",
                password="example-private-password",
            )

            with self.assertRaisesRegex(FirebaseUploadError, "capture_id"):
                upload_dashboard_manifest(
                    manifest_path,
                    config,
                    request_function=fake_request,
                )

        self.assertEqual(calls, [])

    def test_rejects_unpinned_authenticated_uid_before_upload(self) -> None:
        run_id = "run_" + "a" * 24
        def fake_request(
            _url: str,
            _method: str,
            _headers: dict[str, str],
            _body: bytes,
            _timeout_seconds: float,
            _operation: str,
        ) -> dict:
            return {"idToken": "id-token", "localId": "wrong-uid"}

        with tempfile.TemporaryDirectory() as temporary_directory:
            runs = Path(temporary_directory) / "outputs" / "dashboard" / "runs"
            runs.mkdir(parents=True)
            manifest_path = runs / "demo.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "runs": [{"run_id": run_id, "status": "complete"}],
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            config = FirebaseUploadConfig(
                api_key="example-public-web-key",
                expected_uid="jetson-device-uid",
                email="device@example.invalid",
                password="example-private-password",
            )

            with self.assertRaisesRegex(FirebaseUploadError, "pinned"):
                upload_dashboard_manifest(
                    manifest_path,
                    config,
                    request_function=fake_request,
                )

    def test_rejects_changed_artifact_bytes_before_storage_upload(self) -> None:
        run_id = "run_" + "a" * 24
        capture_id = "cap_" + "b" * 32
        calls: list[str] = []

        def fake_request(
            url: str,
            _method: str,
            _headers: dict[str, str],
            _body: bytes,
            _timeout_seconds: float,
            _operation: str,
        ) -> dict:
            calls.append(url)
            return {"idToken": "id-token", "localId": "jetson-device-uid"}

        with tempfile.TemporaryDirectory() as temporary_directory:
            outputs = Path(temporary_directory) / "outputs"
            media = outputs / "dashboard" / "media" / run_id
            runs = outputs / "dashboard" / "runs"
            media.mkdir(parents=True)
            runs.mkdir(parents=True)
            object_key = f"dashboard/media/{run_id}/{capture_id}_crack.png"
            (media / f"{capture_id}_crack.png").write_bytes(
                b"\x89PNG\r\n\x1a\nchanged"
            )
            manifest_path = runs / "demo.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "runs": [{"run_id": run_id, "status": "complete"}],
                        "artifacts": [
                            {
                                "capture_id": capture_id,
                                "artifact_type": "crack_preview",
                                "object_key": object_key,
                                "media_type": "image/png",
                                "sha256": hashlib.sha256(b"original").hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            config = FirebaseUploadConfig(
                api_key="example-public-web-key",
                expected_uid="jetson-device-uid",
                email="device@example.invalid",
                password="example-private-password",
            )

            with self.assertRaisesRegex(FirebaseUploadError, "SHA-256 mismatch"):
                upload_dashboard_manifest(
                    manifest_path,
                    config,
                    request_function=fake_request,
                )

        self.assertEqual(len(calls), 1, "only authentication may occur before rejection")


if __name__ == "__main__":
    unittest.main()
