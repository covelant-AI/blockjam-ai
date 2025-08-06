from utils.bucket import download_file_from_firebase

def test_download_file_from_firebase():
    download_file_from_firebase(
        "https://huggingface.co/AmNat789/CovSports/resolve/main/PTTKrakow.mp4",
        "input_videos/test.mp4"
    )