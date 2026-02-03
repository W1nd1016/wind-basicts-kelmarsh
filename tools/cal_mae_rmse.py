import json

with open("data/wind_scada_cerra_4y_resample/meta.json", "r") as f:
    meta = json.load(f)

y_sd = meta["y_sd"]   # 训练集功率标准差 (kW)

mae_norm = 0.3184
rmse_norm = 0.4449

mae_kw  = mae_norm  * y_sd
rmse_kw = rmse_norm * y_sd

print("sd in kW :", y_sd)
print("MAE in kW :", mae_kw)
print("RMSE in kW:", rmse_kw)