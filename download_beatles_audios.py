import os
import h5py
import json
import yt_dlp
import librosa
from harmony.data.utils import download_audio


if __name__ == '__main__':
    with open('/storage/datasets/thiago.poppe/beatles/the_beatles.json', 'r') as fp:
        data = json.load(fp)

    youtube_urls = set()
    for song in data.values():
        youtube_urls.add(song['youtube']['url'])

    for idx, url in enumerate(youtube_urls):
        try:
            info = download_audio(url)
            wav, sr = librosa.load('audio.wav', sr=44100)

            with h5py.File('/storage/datasets/thiago.poppe/beatles/beatles.h5', 'a') as fp:
                if info['id'] not in fp.keys():
                    fp.create_dataset(info['id'], data=wav, compression='gzip')
                    print(f"Progress {idx+1}/{len(youtube_urls)} => Downloaded and saved {info['id']}\n")
                else:
                    print(f"Progress {idx+1}/{len(youtube_urls)} => Song {info['id']} saved\n")

        except yt_dlp.utils.DownloadError as err:
            print(f'\n{err}\n[warning] Skipping {url}\n')

    # Removing unwanted files
    os.remove('audio.wav')
