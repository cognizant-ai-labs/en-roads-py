"""
Setup script to load the En-ROADS SDK from the S3 bucket. This is used for app deployment and testing.
Note: This script requires access to the private S3 bucket where the SDK is stored. If you would like access to the
SDK to run the full project, see the README in order to contact a member of Project Resilience.
"""
import os
import zipfile

import boto3


def main():
    """
    Downloads En-ROADS SDK from the S3 bucket and extracts it.
    If the sdk already exists, we do nothing.
    If we already have the zip file but no SDK, we just extract the zip file.
    """
    sdk_name = "en-roads-sdk-v24.6.0-beta1"
    sdk_path = "enroadspy/"
    zip_path = sdk_path + "/" + sdk_name + ".zip"

    if os.path.exists(sdk_path + "/" + sdk_name):
        print("SDK already exists.")
        return

    if not os.path.exists(zip_path):
        s3 = boto3.client('s3')
        s3.download_file("ai-for-good", sdk_name + ".zip", zip_path)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(sdk_path)


if __name__ == "__main__":
    main()
