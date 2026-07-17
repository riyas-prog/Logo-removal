from group_manager import *

videos = [

    create_fingerprint(
        "video1.mp4",
        432,
        768,
        30,
        "top_left",
        190,
        70
    ),

    create_fingerprint(
        "video2.mp4",
        432,
        768,
        30,
        "top_left",
        188,
        72
    ),

    create_fingerprint(
        "video3.mp4",
        432,
        768,
        30,
        "top_right",
        190,
        70
    ),

    create_fingerprint(
        "video4.mp4",
        720,
        1280,
        30,
        "top_left",
        190,
        70
    ),

]

groups = group_videos(videos)

for key, items in groups.items():

    print("=" * 60)
    print("GROUP:", key)

    for item in items:
        print("  -", item.filename)