from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.data.utils import ValSplitMode
from anomalib.deploy import ExportType
from anomalib.engine import Engine
from anomalib.models import Padim


def export():
    datamodule = MVTec(
        root=r"F:\demo\mvtec_anomaly_detection",
        category="bottle",
        task=TaskType.CLASSIFICATION,
        val_split_mode=ValSplitMode.SYNTHETIC,  # synthetically generate validation data
        image_size=(256, 256),
        val_split_ratio=0.2,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=4,
    )

    datamodule.setup()

    # Model & engine
    model = Padim()
    engine = Engine()

    # Exporting model to OpenVINO
    openvino_model_path = engine.export(
        model=model,
        export_type=ExportType.OPENVINO,
        export_root=r"F:\demo\anomalydetection\models",  # OpenVINO model will be saved here
        ckpt_path=r"F:\demo\anomalydetection\results\Padim\MVTec\bottle\v0\weights\lightning\model.ckpt",
    )
    print(f"OpenVINO Model saved to {str(openvino_model_path)}")


if __name__ == "__main__":
    export()
