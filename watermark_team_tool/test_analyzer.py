from analyzer import *

video = VideoAnalysis(
    filename="RoyalMatch.mp4",
    filepath="videos/RoyalMatch.mp4",

    width=432,
    height=768,

    fps=30,
    frame_count=900,
    duration=30,

    detected_corner="top_left",

    logo_x=0,
    logo_y=0,
    logo_w=190,
    logo_h=70,

    detection_score=98.7
)

print(video)