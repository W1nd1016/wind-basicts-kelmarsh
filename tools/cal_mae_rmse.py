import json

with open("data/wind/meta.json", "r") as f:
    meta = json.load(f)

y_sd = meta["y_sd"]   # 训练集功率标准差 (kW)

mae_norm = 0.4676
rmse_norm = 0.6444

mae_kw  = mae_norm  * y_sd
rmse_kw = rmse_norm * y_sd

print("sd in kW :", y_sd)
print("MAE in kW :", mae_kw)
print("RMSE in kW:", rmse_kw)