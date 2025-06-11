# !/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2018.
# @Author : 张明阳
# @Email : mingyang.zhang@ushow.media



import random
import string

def random_name(prefix, length=6):
    return prefix + ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def generate_dataset(
    filename="163_music_playlist.txt",
    num_playlists=1000,
    songs_per_playlist_range=(10, 30),
    song_vocab_size=5000,
    artists_count=1000
):
    # 生成歌曲字典：song_id -> (song_name, artist, popularity)
    song_dict = {}
    for song_id in range(1000, 1000 + song_vocab_size):
        song_name = random_name("Song_")
        artist_id = random.randint(1, artists_count)
        artist_name = f"Artist_{artist_id}"
        popularity = random.randint(50, 100)
        song_dict[str(song_id)] = (song_name, artist_name, popularity)

    with open(filename, "w", encoding="utf-8") as f:
        for i in range(1, num_playlists + 1):
            playlist_id = f"playlist_{i:05d}"
            num_songs = random.randint(*songs_per_playlist_range)
            song_ids = random.sample(list(song_dict.keys()), num_songs)

            playlist_line = [playlist_id]
            for song_id in song_ids:
                song_name, artist, pop = song_dict[song_id]
                song_info = f"{song_id}:::{song_name}:::{artist}:::{pop}"
                playlist_line.append(song_info)

            f.write("\t".join(playlist_line) + "\n")

    print(f"✅ Finished generating {num_playlists} playlists into `{filename}`.")

# 调用生成函数
generate_dataset(
    filename="../../../data_files/163_music_playlist.txt",
    num_playlists=1000,
    songs_per_playlist_range=(10, 30),
    song_vocab_size=5000,
    artists_count=1000
)

