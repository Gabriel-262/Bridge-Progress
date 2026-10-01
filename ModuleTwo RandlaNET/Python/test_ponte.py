import torch
import open3d.ml.torch as ml3d
from dataset_ponte import PonteDataset

# 1. Carrega o nosso Dataset
dataset = PonteDataset(
    dataset_path="/home/joserasj/RandlaNET/DATASET/Ponte_Custom3D"
)

# 2. Configura a Rede Neural exatamente como no treino
model = ml3d.models.RandLANet(
    num_points=40960,  
    num_classes=2,     
    in_channels=6,     
    ignored_label_inds=[]
)

# 3. Monta o Pipeline (Não precisamos mais de batch_size ou learning rate aqui)
pipeline = ml3d.pipelines.SemanticSegmentation(
    model=model,
    dataset=dataset,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# 4. CARREGA O CÉREBRO TREINADO (A Época 50)
caminho_dos_pesos = "./logs_ponte/RandLANet_PonteDataset_torch/checkpoint/ckpt_00050.pth"
print(f"Carregando o modelo treinado em: {caminho_dos_pesos}")

pipeline.load_ckpt(ckpt_path=caminho_dos_pesos)
print("Cérebro carregado com sucesso!")

# 5. Inicia a prova final!
print("Iniciando avaliação na GPU...")
pipeline.run_test()