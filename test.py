from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Padim


def train():
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

    # Test the model
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=r"F:\demo\anomalydetection\results\Padim\MVTec\bottle\v0\weights\lightning\model.ckpt",
    )
    print(test_results)


if __name__ == "__main__":
    train()
