import pandas as pd

#Rename synthetic dataset CSV columns from codes to descriptive names

IN_CSV  = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\lit_synth_5s_states.csv"
OUT_CSV = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\lit_synth_5s_states_named.csv"

CODE_TO_SHORT_BASE = {
"A0":"Microwave_Standby",
"B0":"LED_Lamp",
"C0":"CRT_Monitor_Standby",
"D0":"LED_Panel",
"E0":"Smoke_Extractor",
"F0":"LED_Monitor_Standby",
"G0":"Phone_Charger_Asus",
"H0":"Soldering_Station",
"I0":"Phone_Charger_Motorola",
"J0":"Universal_Charger",
"K0":"Fan_3Speed",
"L0":"Resistor_200ohm",
"M0":"AC_Adapter_Sony",
"N0":"Incandescent_Lamp",
"O0":"Impact_Drill",
"P0":"Impact_Drill",
"Q0":"Oil_Heater",
"R0":"Oil_Heater",
"S0":"Microwave_On",
"T0":"Fan_Heater",
"U0":"Hair_Dryer_1900W",
"V0":"Hair_Dryer_1900W",
"W0":"Hair_Dryer_2000W",
"X0":"Hair_Dryer_2000W",
"Y0":"Hair_Dryer_2100W",
"Z0":"Hair_Dryer_2100W",
}

df = pd.read_csv(IN_CSV)

rename_map = {}
for c in df.columns:
    if c.startswith("y_"):
        code = c[2:]  # A0, B0, ...
        base = CODE_TO_SHORT_BASE.get(code, code)
        # duplicate-safe: append the code
        rename_map[c] = f"y_{base}_{code}"

df = df.rename(columns=rename_map)
df.to_csv(OUT_CSV, index=False)

print("Wrote:", OUT_CSV)
print("Example y cols:", [c for c in df.columns if c.startswith("y_")][:12])


# import json

# OUT_JSON = r"C:\Users\ASUS\Desktop\Projects\ML Project\Dataset\Exports\lit_code_to_official_name.json"

# CODE_TO_OFFICIAL = {
# "A0":"Microwave, Consul 17 liters CMS18BBHNA,127V 1200W - standby 4.5W",
# "B0":"LED Lamp, Tashibra TKl 06, 127V 6W",
# "C0":"CRT Monitor, Sony CPD-17SF1, 127V 216W - standby 10W",
# "D0":"LED Panel, Citiiaqua DL-500, 127V 13W",
# "E0":"Soldering Smoke Extractor, Toyo TS-153, 127V 23W",
# "F0":"LED monitor, AOC m2470swd2, 127V 26W, standby 0.5W",
# "G0":"Phone Charger, Asus AD2037020, 110V-240V 38W",
# "H0":"Soldering Station, Weller WLC100, 127V 40W",
# "I0":"Phone Charger, Motorola SA-390M, 127V 50W",
# "J0":"Universal Charger, LVSUN LS-PAB70, 100V-240V 70W",
# "K0":"3 Speed Fan, Mondial V-45, 127V 80W",
# "L0":"Resistor, Ohmtec 200ohms, 400W",
# "M0":"AC Adapter Charger, Sony PCG-61112L, 127V 92W",
# "N0":"Incandescent Lamp, Osram Centra A CL 100, 127V 100W",
# "O0":"2 Speed Impact Drill, Bosch 47CV, 127V 350W",
# "P0":"2 Speed Impact Drill, Bosch 47CV, 127V 350W",
# "Q0":"Oil Heater, Pelonis NYLA-7, 127V 1500W",
# "R0":"Oil Heater, Pelonis NYLA-7, 127V 1500W",
# "S0":"Microwave, Consul CMS18BBHNA, 127V 1200W",
# "T0":"Fan Heater, Nilko NK565, 127V 1500W",
# "U0":"Hair Dryer, GA.MA Italy Eleganza 2200, 127V 1900W",
# "V0":"Hair Dryer, GA.MA Italy Eleganza 2200, 127V 1900W",
# "W0":"Hair Dryer, Super 4.0 SL-S04, 127V 2000W",
# "X0":"Hair Dryer, Super 4.0 SL-S04, 127V 2000W",
# "Y0":"Hair Dryer, Parlux 330 BR/1, 127V 2100W",
# "Z0":"Hair Dryer, Parlux 330 BR/1, 127V 2100W",
# }

# with open(OUT_JSON, "w", encoding="utf-8") as f:
#     json.dump(CODE_TO_OFFICIAL, f, indent=2)

# print("Wrote:", OUT_JSON)
