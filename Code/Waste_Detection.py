import argparse
from pathlib import Path
import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.utils.files import increment_path

def run(weights="yolov8n.pt", source="test.mp4", view_img=False, save_img=False, exist_ok=False, device='cpu'):
    """
    Run object detection on a video using YOLOv8 and SAHI.

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        device (str): Device to use for inference (e.g., cpu, cuda:0, cuda:1).
    """

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Initialize detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",  # Adjust this if using a different model type
        model_path=weights,
        confidence_threshold=0.8,
        device=device
    )

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Output setup
    save_dir = increment_path(Path("ultralytics_results_with_sahi") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (frame_width, frame_height))
    print("Results saved to ", save_dir)

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break

        results = get_sliced_prediction(
            frame, detection_model, slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2
        )
        object_prediction_list = results.object_prediction_list

        for prediction in object_prediction_list:
            bbox = prediction.bbox
            cls = prediction.category.name
            cv2.rectangle(frame, (int(bbox.minx), int(bbox.miny)), (int(bbox.maxx), int(bbox.maxy)), (56, 56, 255), 2)
            label = str(cls)
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            cv2.rectangle(
                frame, (int(bbox.minx), int(bbox.miny) - t_size[1] - 3), (int(bbox.minx) + t_size[0], int(bbox.miny) + 3), (56, 56, 255), -1
            )
            cv2.putText(
                frame, label, (int(bbox.minx), int(bbox.miny) - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA
            )

        if save_img:
            video_writer.write(frame)

        # Removed the cv2.imshow() and cv2.waitKey() lines
        # if view_img:
        #     cv2.imshow(Path(source).stem, frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    video_writer.release()
    videocapture.release()
    #cv2.destroyAllWindows()

def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n", help="initial weights path")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for inference (e.g., cpu, cuda:0, cuda:1)')
    return parser.parse_args()

def main(opt):
    """Main function."""
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
