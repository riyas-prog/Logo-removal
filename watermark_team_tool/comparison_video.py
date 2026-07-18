import cv2


def create_side_by_side(original_path, cleaned_path, output_path):
    cap1 = cv2.VideoCapture(original_path)
    cap2 = cv2.VideoCapture(cleaned_path)

    fps = cap1.get(cv2.CAP_PROP_FPS)

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width * 2, height)
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        cv2.putText(
            frame1,
            "ORIGINAL",
            (20, 40),
            font,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame2,
            "LOGO REMOVED",
            (20, 40),
            font,
            1,
            (0, 255, 0),
            2
        )

        combined = cv2.hconcat([frame1, frame2])

        writer.write(combined)

    cap1.release()
    cap2.release()
    writer.release()

    print("Comparison video created:", output_path)