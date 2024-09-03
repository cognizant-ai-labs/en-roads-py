"""
Setup script to download the En-ROADS SDK. This is used for app deployment and testing.
"""
import os
import zipfile

import requests


def main():
    """
    Downloads en-roads sdk and extracts it.
    If the sdk already exists, we do nothing.
    If we already have the zip file but no SDK, we just extract the zip file.
    """
    zip_path = "enroadspy/en-roads-sdk-v24.6.0-beta1.zip"
    sdk_path = "enroadspy/en-roads-sdk-v24.6.0-beta1"

    if os.path.exists(sdk_path):
        print("SDK already exists.")
        return

    if not os.path.exists(zip_path):
        url = "https://en-roads.dev.climateinteractive.org/branch/release/24.6.0/en-roads-sdk-v24.6.0-beta1.zip"

        username = os.getenv("ENROADS_ID")
        password = os.getenv("ENROADS_PASSWORD")
        assert username is not None and password is not None, \
            "Please set the ENROADS_ID and ENROADS_PASSWORD environment variables."

        r = requests.get(url, auth=(username, password), timeout=60)

        if r.status_code == 200:
            with open(zip_path, "wb") as out:
                for bits in r.iter_content():
                    out.write(bits)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(sdk_path)


if __name__ == "__main__":
    main()
