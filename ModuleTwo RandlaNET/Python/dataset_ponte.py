import glob
import numpy as np
from plyfile import PlyData

from open3d._ml3d.datasets.base_dataset import BaseDataset
from open3d._ml3d.datasets.base_dataset import BaseDatasetSplit

class PonteDataset(BaseDataset):
    def __init__(self, dataset_path, **kwargs):
        super().__init__(
            dataset_path=dataset_path,
            name="PonteDataset",
            **kwargs
        )

        # 1. Definindo as 2 classes da nossa ponte
        self.label_to_names = {
            0: "floor",
            1: "column"
        }

        self.num_classes = 2
        self.cfg.use_cache = False

        # Apontando para as pastas corretas onde você deve colocar os arquivos .ply
        self.train_files = sorted(glob.glob(dataset_path + "/train/*.ply"))
        self.val_files = sorted(glob.glob(dataset_path + "/val/*.ply"))

    @staticmethod
    def get_label_to_names():
        return {
            0: "floor",
            1: "column"
        }

    def is_tested(self, attr):
        return False

    def save_test_result(self, results, attr):
        pass

    def get_split_list(self, split):
        if split == "train":
            return self.train_files
        elif split in ["test", "val", "validation"]:
            return self.val_files
        else:
            raise ValueError("split inválido")

    def get_split(self, split):
        if split == "train":
            files = self.train_files
        elif split in ["test", "val", "validation"]:
            files = self.val_files
        else:
            raise ValueError("split inválido")

        return PonteSplit(self, split, files)


class PonteSplit(BaseDatasetSplit):
    def __init__(self, dataset, split, files):
        self.files = files
        super().__init__(dataset, split=split)

    def __len__(self):
        # TRUQUE: Multiplicamos por 100 para forçar a IA a treinar mais vezes
        return len(self.files) * 100

    def get_data(self, idx):
        # TRUQUE: Usamos o (%) para ele ler sempre o mesmo arquivo sem dar erro
        file = self.files[idx % len(self.files)]
        
        print(f"[{self.split}] Carregando arquivo: {file}")
        
        ply = PlyData.read(file)
        vertex = ply["vertex"].data

        points = np.vstack([
            vertex["x"],
            vertex["y"],
            vertex["z"]
        ]).T.astype(np.float32)

        # A nossa correção de coordenadas continua aqui
        points = points - np.min(points, axis=0)

        feat = np.vstack([
            vertex["red"],
            vertex["green"],
            vertex["blue"]
        ]).T.astype(np.float32) / 255.0

        labels = np.array(vertex["label"]).astype(np.int32)

        return {
            "point": points,
            "feat": feat,
            "label": labels
        }

    def get_attr(self, idx):
        # TRUQUE AQUI TAMBÉM: usar o módulo para ele repetir o arquivo corretamente
        file = self.files[idx % len(self.files)]
        
        name = file.split("/")[-1].replace(".ply", "")
        return {
            "idx": idx,
            "name": name,
            "path": file,
            "split": self.split,
            "num_points": len(self.files) 
        }