from typing import Any

from anomalib import TaskType
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
from anomalib.utils.visualization.image import ImageVisualizer, VisualizationMode
from PIL import Image


def get_predictions(
    image_path: str,
    metadata_path: str,
    model_path: str,
) -> Any:
    image = read_image(path=image_path)
    inferencer = OpenVINOInferencer(
        path=model_path,  # Path to the OpenVINO IR model.
        metadata=metadata_path,  # Path to the metadata file.
        device="CPU",
    )
    predictions = inferencer.predict(image=image)
    return predictions


def visualize_anomaly(predictions: Any) -> Image:
    visualizer = ImageVisualizer(
        mode=VisualizationMode.FULL, task=TaskType.CLASSIFICATION
    )
    output_image = visualizer.visualize_image(predictions)
    return Image.fromarray(output_image)


if __name__ == "__main__":
    predictions = get_predictions(
        r"F:\demo\mvtec_anomaly_detection\bottle\test\broken_large\000.png",
        r"F:\demo\anomalydetection\models\weights\openvino\metadata.json",
        r"F:\demo\anomalydetection\models\weights\openvino\model.bin",
    )
    out_image = visualize_anomaly(predictions)
    out_image.save("result.png")
