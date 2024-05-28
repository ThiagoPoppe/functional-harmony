import os
import time
import json
import yt_dlp
import librosa
import zipfile

import numpy as np
import soundfile as sf

from collections import defaultdict


def download_audio(url):
    ydl_opts = {
        'quiet': True,
        'nocache': True,
        'noprogress': True,
        'noplaylist': True,
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': 'audio.%(ext)s'
    }

    sleep_time = np.random.choice(range(5, 11))
    time.sleep(sleep_time)

    print('Downloading url:', url)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

    return info


if __name__ == '__main__':
    basepath = '/storage/datasets/harmony_recognition/HookTheory/'
    datapath = ospj(basepath, 'hooktheory.json')

    with open(datapath, 'r') as fp:
        data = json.load(fp)

    # Grouping song ids by split and youtube url
    grouped_data = defaultdict(lambda: defaultdict(list))

    for song_id, song in data.items():
        split = song['split'].lower()
        youtube_url = song['youtube']['url']
        grouped_data[split][youtube_url].append(song_id)

    grouped_data['validation'] = grouped_data.pop('valid')

    print('Number of data entries:', len(data))
    print('Grouped data keys:', grouped_data.keys())

    # Creating file to save processed URLs
    with open('processed_urls.txt', 'a') as fp:
        pass

    # Reading already processed songs
    with open('processed_urls.txt', 'r') as fp:
        processed_urls = set([line.strip() for line in fp.readlines()])

    for split, grouped_urls in grouped_data.items():
        print(f'\n[log] Processing {split} URLs\n')

        url_counter = 0
        total_urls = len(grouped_urls)

        for youtube_url, song_ids in grouped_urls.items():
            if youtube_url in processed_urls:
                url_counter += 1
                print(f'[log] Skipping already processed URL {youtube_url}\n')
                continue

            try:
                download_audio(youtube_url)
                wave, sr = librosa.load('audio.wav', sr=SAMPLING_RATE)
            except yt_dlp.utils.DownloadError as err:
                print(err)
                print(f'[warning] Skipping {youtube_url} and song ids {song_ids}\n')
                url_counter += 1
                continue

            for song_id in song_ids:
                if ('USER_ALIGNMENT' not in data[song_id]['tags']) and ('REFINED_ALIGNMENT' not in data[song_id]['tags']):
                    print(f'[warning] Skipping {song_id} due to no alignment found\n')
                    continue

                duration = data[song_id]['youtube']['duration']
                alignment_type = 'refined' if data[song_id]['alignment']['refined'] else 'user'

                start_time = data[song_id]['alignment'][alignment_type]['times'][0]
                end_time = data[song_id]['alignment'][alignment_type]['times'][-1]

                start = librosa.time_to_samples(start_time, sr=sr)
                end = librosa.time_to_samples(end_time, sr=sr)

                song = data[song_id]['hooktheory']['song']
                artist = data[song_id]['hooktheory']['artist']
                section = data[song_id]['hooktheory']['section']

                with zipfile.ZipFile(ospj(basepath, f'dataset/audios/{split}.zip'), 'a') as zf:
                    sf.write('section_wave.wav', wave[start:end], sr, subtype='PCM_24')
                    zf.write('section_wave.wav', arcname=f'{song_id}.wav')

            with open('processed_urls.txt', 'a') as fp:
                url_counter += 1
                processed_urls.add(youtube_url)
                fp.write(f'{youtube_url}\n')
                print(f'\n[log] Processed URLs: {url_counter}/{total_urls}\n')

            if url_counter == 250:
                print('[log] Long sleep after 250 requests')
                time.sleep(300)  # waits 5min after 250 requests

    # Removing temporary files
    os.remove('audio.wav')
    os.remove('section_wave.wav')
