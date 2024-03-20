import requests
import hashlib
from typing import Dict, Any


class Civit:
    def __init__(self, base_url="https://civitai.com/api/v1/"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}

    def _read_file(self, filename):
        try:
            with open(filename, "rb") as file:
                file.seek(0x100000)
                return file.read(0x10000)
        except FileNotFoundError:
            return b"NOFILE"
        except Exception:
            return b"NOHASH"

    def model_hash(self, filename):
        """old hash that only looks at a small part of the file and is prone to collisions"""
        file_content = self._read_file(filename)
        m = hashlib.sha256()
        m.update(file_content)
        shorthash = m.hexdigest()[0:8]
        return shorthash

    def get_models_by_hash(self, hash):
        url = f"{self.base_url}model-versions/by-hash/{hash}"
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                print("Error: Model Not Found on civit.ai")
                return {}
            elif response.status_code == 503:
                print("Error: Civit.ai Service Currently Unavailable")
                return {}
            else:
                print(f"HTTP Error: {e}")
                return {}
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return {}

    def get_keywords(self, model):
        keywords = model.get("trainedWords", ["No Keywords for LoRA"])
        return keywords
