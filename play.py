from models import build_diag_gauss_policy



model = build_diag_gauss_policy(1, [64, 64], 10)


for layer in model.children():
    print(layer.parameters())
    print(layer)

