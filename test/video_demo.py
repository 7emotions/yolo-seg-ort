from yolo_seg_ort import YOLOSeg, Results
import cv2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    onnx_path = r"../models/yolo11n-seg.onnx"

    try:
        model = YOLOSeg(onnx_model=onnx_path)
        while True:
            ret, frame = cap.read()
            result: list[Results] = model(frame)
            if result:
                result[0].show()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"加载 ONNX 模型时发生错误：{e}")
        print("请确保您有一个有效的 ONNX 模型文件路径，例如 './best.onnx'")

    cap.release()
    cv2.destroyAllWindows()
