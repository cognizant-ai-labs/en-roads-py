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
    sdk_path = "enroadspy/"

    if os.path.exists(sdk_path + "en-roads-sdk-v24.6.0-beta1"):
        print("SDK already exists.")
        return

    if not os.path.exists(zip_path):
        url = os.getenv("ENROADS_URL")
        username = os.getenv("ENROADS_ID")
        password = os.getenv("ENROADS_PASSWORD")
        assert url is not None, \
            "Please set the ENROADS_URL environment variable."
        assert len(url) == 96, \
            f"ENROADS_URL is not the correct length, it is {len(url)} characters instead of 96."
        assert username is not None and password is not None, \
            "Please set the ENROADS_ID and ENROADS_PASSWORD environment variables. \
            To get access to them go to https://en-roads.climateinteractive.org/ and sign up."

        r = requests.get(url, auth=(username, password), timeout=60)

        if r.status_code == 200:
            with open(zip_path, "wb") as out:
                for bits in r.iter_content():
                    out.write(bits)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(sdk_path)


if __name__ == "__main__":
    main()
