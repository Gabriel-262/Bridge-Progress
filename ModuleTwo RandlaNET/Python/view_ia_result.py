import numpy as np
import open3d as o3d
import open3d.ml.torch as ml3d
from dataset_ponte import PonteDataset
from plyfile import PlyData

print("1. Carregando as configurações da IA...")
dataset = PonteDataset(dataset_path="/home/joserasj/Módulos/ModuleTwo RandlaNET/DATASET/Ponte_Custom3D")

model = ml3d.models.RandLANet(
    num_points=40960,  
    num_classes=2,     
    in_channels=6,     
    ignored_label_inds=[]
)

pipeline = ml3d.pipelines.SemanticSegmentation(
    model=model,
    dataset=dataset,
    device="cuda"
)

# Carrega os pesos da Época 50
pipeline.load_ckpt("./logs_ponte/RandLANet_PonteDataset_torch/checkpoint/ckpt_00050.pth")

print("2. Carregando a nuvem de pontos da ponte...")
#ply_path = "/home/joserasj/Módulos/ModuleTwo RandlaNET/DATASET/Ponte_Custom3D/test/ponte.ply"
ply_path = "/home/joserasj/Módulos/ModuleTwo RandlaNET/DATASET/D19/ponte.ply"

ply = PlyData.read(ply_path)
vertex = ply["vertex"].data

points = np.vstack([vertex["x"], vertex["y"], vertex["z"]]).T.astype(np.float32)
points = points - np.min(points, axis=0)
feat = np.vstack([vertex["red"], vertex["green"], vertex["blue"]]).T.astype(np.float32) / 255.0

# : Criamos um array de zeros do mesmo tamanho da nuvem
dummy_labels = np.zeros(points.shape[0], dtype=np.int32)

# O dicionário que a IA exige para fazer a predição agora está completo
data = {
    "point": points,
    "feat": feat,
    "label": dummy_labels  # <--- Adicionamos o "falso gabarito"
}

print("3. Pedindo para a IA adivinhar o que é Pilar e o que é Chão...")
# A IA vai processar os pontos e devolver uma lista de predições (0 ou 1)
resultados = pipeline.run_inference(data)
predicoes = resultados['predict_labels']

print("4. Montando o visualizador 3D...")
# Criamos uma paleta vazia
cores_da_ia = np.zeros_like(points)

# Pintamos de Azul (Chão) onde a IA previu 0
cores_da_ia[predicoes == 0] = [0.0, 0.6, 1.0] 

# Pintamos de Vermelho (Pilar) onde a IA previu 1
cores_da_ia[predicoes == 1] = [1.0, 0.0, 0.0] 

# Criamos a Nuvem de Pontos final para visualização
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(cores_da_ia)

print("\nAbrindo janela do Open3D! Verifique como a IA segmentou a sua ponte.")
o3d.visualization.draw_geometries([pcd])